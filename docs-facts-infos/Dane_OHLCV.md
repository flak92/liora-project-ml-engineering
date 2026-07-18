# Dane OHLCV — źródła, przetwarzanie i weryfikacja

Ten dokument opisuje **dane** w projekcie: skąd pochodzą, jak są przetwarzane w cechy
i etykiety, oraz co potwierdziła weryfikacja na realnych zapieczętowanych barach.
Wszystkie fakty mają źródło (plik:linia / pomiar). Stan: 2026-07-18, epoka
`2026-07-golden-v4`, gałąź `Stable_Presentable_Version`.

## 1. Źródła danych

Dwa surowe strumienie OHLCV, po jednym na model:

- **XGB → bary 1-godzinne (1h)** — S&P 500, zapieczętowane per asset w
  `<T>_ohlcv_1h.parquet`.
- **LSTM → bary dzienne (1d)** — S&P 500, w committed `data/sp500_1d.duckdb`.

**Interwały 1d i 1w w plikach XGB to NIE trzecie źródło danych** — są **wyliczane z tego
samego strumienia 1h** jako kontekst wielointerwałowy (`CONTEXT_TIMEFRAMES = ("1d","1w")`,
`src/xgb/pipeline.py`). Bary godzinowe są agregowane w świece dzienne i tygodniowe, a
następnie **przyczynowo** rzutowane z powrotem na oś godzinową (merge_asof po zamknięciu
ukończonego dnia/tygodnia). Stąd cechy z sufiksami `_1d`/`_1w` (np. `volume_z_score_20_1w`)
i cechy zestrojenia między interwałami. Snapshot `<T>_ohlcv_1w.parquet` w folderach
roboczych to zmaterializowana ta sama agregacja (do reprodukcji), nie osobne wejście.

## 2. Jak dane są przetwarzane

Łańcuch: **OHLCV → cechy / sekwencje → etykiety triple-barrier (Train) → modele per asset**.

- **Bramki jakości (QC) przy wczytaniu** — monotoniczność czasu, brak duplikatów, sanity
  OHLC (high ≥ max(open,close), low ≤ min(open,close)), ceny > 0, wolumen ≥ 0
  (`src/xgb/pipeline.py` layer4, `src/lstm/pipeline.py` load_bars).
- **Cechy kroczące** — okna po indeksie bara (rolling(20) itd.), spójne między Train i OOS.
- **Agregacja 1h→1d/1w** — open=pierwszy bar, high=max, low=min, close=ostatni, volume=suma;
  grupowanie po sesji ET / tygodniu ISO; kontekst dostępny dopiero po ukończeniu okresu.
- **Etykiety** — asymetryczny triple-barrier ATR (TP = 2×ATR, SL = 1×ATR); decyzja na
  close[t0], wejście fillowane na otwarciu następnej sesji; purge/embargo trzyma horyzont
  etykiet Train przed OOS. Model podejmuje **wyłącznie decyzję ENTRY** przy zamrożonym
  kontrakcie TP/SL — TP i SL to bariery mechaniczne, nie decyzja modelu.
- **Zero imputacji** — braki nie są wypełniane; wiersze z niefinitną cechą CORE są
  wykluczane z kandydatów, cechy opcjonalne XGB mogą nieść NaN (gałąź missing drzewa),
  okna LSTM z niepełnymi danymi są pomijane (bez fabrykowania wartości).

## 3. Co potwierdziła weryfikacja (rdzeń przetwarzania jest zdrowy)

Dogłębna weryfikacja (4 niezależnych recenzentów, pomiary na AAPL, NVDA, ABNB, COIN, CEG,
GEHC, MSFT, VLTO + skany całego universum 498 tickerów):

| Obszar danych | Wynik |
|---|---|
| Agregacja 1h→1d/1w + przyczynowość | ✅ w pełni poprawne |
| Bramki QC per-bar | ✅ działają (skan 498 parquetów czysty) |
| Cechy sesyjne / okna kroczące wokół gapów | ✅ poprawne i spójne Train↔OOS |
| NaN / warmup / eligibility / okna LSTM | ✅ zero NaN w rdzeniu, zero imputacji |
| Mechanika transakcji przez gapy | ✅ uczciwa (fill na realnym otwarciu) |

Dowody:
- **Agregacja bit-w-bit**: ręczna rekonstrukcja świec 1d/1w = **0.0 max błędu** vs kod i vs
  zapieczętowane parquety (5 tickerów, w tym IPO ABNB/COIN/GEHC), z poprawną obsługą sesji
  częściowych i skróconych świąt.
- **Przyczynowość udowodniona trunkacją**: obcięcie wejścia na granicy ukończonego tygodnia
  zostawia WSZYSTKIE kolumny `_1d/_1w`/multi_tf bitowo identyczne — **zero wycieku w przód**;
  środa w środku tygodnia bierze poprzedni okres.
- **QC działa**: preparowane złe ramki (duplikat, high<low, cena≤0, ujemny wolumen) rzucają
  RuntimeError; skan 498 parquetów = 0 duplikatów / 0 niemonotonicznych / 0 naruszeń OHLC /
  0 ujemnego wolumenu.
- **NaN / warmup**: wiodące NaN cech = dokładnie rozmiar okna; rdzeń macierzy treningowej
  = 0 NaN / 0 Inf; IPO-era warmup nie przecieka (VLTO: 49 barów odfiltrowanych); okna LSTM
  odpadające na bramce skończoności są WYŁĄCZNIE warmupowe (zero dziur w środku historii);
  księgowość `n + n_nan == kandydaci` zgadza się co do wiersza.
- **Transakcje przez gapy**: decyzja na close[t0], fill na PRAWDZIWYM otwarciu następnej
  sesji ze slippage (gap nocny/weekendowy poniesiony, nie fantazyjna cena); purge/embargo
  `t0+H+embargo ≤ oos_start` zweryfikowane; wagi unikalności w (0,1].

## 4. Obszar do domknięcia — dostosowanie do splitów (błąd do poprawy)

Weryfikacja wskazała **jeden realny błąd w danych wejściowych do poprawienia**: bary są w
wersji **nieskorygowanej o splity** (`CORP_ACTIONS_POLICY='deferred'` jest zwalidowana w
kodzie, samo dostosowanie do wykonania jako osobny krok). Opisujemy to jawnie i planujemy
korektę — potwierdzone własnym pomiarem na committed `data/results.db` + surowych barach:

- **Zasięg**: **79/498 tickerów (16%)** ma gap splitowy w historii — **31 w OOS, 58 w Train**.
- **Przykład NVDA**: surowy gap overnight −75,2% (split 4:1, 2021-07-20, Train) i −90,1%
  (split 10:1, 2024-06-10, OOS) — to zmiany czysto techniczne (podział akcji), nie ruch ceny.
- **Benchmark HODL na cenach surowych** pokazuje dla tych tickerów wartości nieskorygowane
  (NVDA `-56,34%`, CMG `-98,56%`, AVGO `-62%`, WMT `-24%`). Po dostosowaniu do splitów
  odpowiadają realnym, w większości dodatnim zwrotom buy-and-hold — korekta sprowadza je do
  prawdziwego HODL, a wraz z nimi statystyki „beats-HODL".
- **Dobra wiadomość co do zasięgu**: dostosowanie poprawia głównie **porównanie z HODL** i
  zwroty 79 tickerów trzymanych przez split; **nie dotyka** mechaniki filtra ENTRY ani
  warstwy interpretacyjnej (oparte na Train, w większości okna poza splitem). Pozostałe 419
  tickerów oraz cała mechanika przetwarzania są potwierdzone jako poprawne.

### Koszt obliczeniowy korekty (16 rdzeni / 30 GiB; z realnych czasów nocy v4)

| Poziom | Co obejmuje | Czas @16c |
|---|---|---|
| **1. Tylko HODL (read-side)** | Przeliczyć benchmark na cenach skorygowanych, poprawić `beats_hodl`/`hodl_return_pct` | **~5–15 min**, zero re-treningu |
| **2. Tylko 79 tickerów splitowych** | Korekta ich barów + re-seal + re-ekstrakcja + HODL uniwersum | **~45–70 min** |
| **3. Pełna epoka v5 (docelowo)** | Korekta całego universum → re-seal → żniwa → interpretacja → results.db → bramki | **~6–7 h / jedna noc ≈ ~100 rdzeniogodzin** |

Skorygowane ceny zmieniają cechy → etykiety → modele, więc pełne domknięcie wymaga re-runu
(poziom 3). Na prezentację **teraz** wystarcza **poziom 1** (~kwadrans, bez re-treningu) +
krótka nota o splitach na Overview/Model Comparison, z oznaczeniem 79 tickerów przy „beats-HODL".

## 5. Drobne domknięcia jakości danych (usprawnienia)

- **Rozszerzenie QC**: obecne bramki sprawdzają sanity per-bar; warto dodać kontrolę
  kompletności sesji / płaskich barów. 3 ogólnorynkowe dni z niepełnymi danymi
  (2021-04-19, 2021-10-25, 2022-03-08; + 2018-05-02/03) zwijają sesję do jednego płaskiego
  bara — **3825 płaskich barów u 475 tickerów (0,4% wszystkich)**, wszystkie w Train, wpływ
  niewielki i ograniczony do wąskiego okna kroczącego. Warta jawnej bramki w v5.
- **Detekcja barier po CLOSE** jest z natury ostrożna po stronie win-rate o ~5 pp (ciaśniejszy
  SL 1×ATR bywa dotykany intra-bar częściej niż TP 2×ATR); mechanizm jest w kodzie — warto
  dopisać kierunek i skalę do `docs/METHODOLOGY.md`.
- **Dostrojenie zdania w dokumencie**: METHODOLOGY.md §2 mówi „Y=1 gdy TP przed SL"; kod
  etykietuje po znaku zwrotu netto (z kosztami i gapem) — kod jest dokładniejszy niż opis.
- Cechy `*_alignment_multi` we wczesnej historii degradują do dostępnych interwałów zamiast
  zostać NaN — przyczynowe i bez wycieku; do udokumentowania jako zamierzone.

## 6. Kolejność prac i sedno przekazu o danych

1. **Korekta splitów** — minimum poziom 1 przed pokazaniem „beats-HODL"; docelowo epoka v5.
2. **Dalej z designem** — sześć stron konsoli Streamlit nad gotowym `data/results.db`
   (jedyny moduł dostępu `app/data.py`), z jawną notą o splitach.
3. **Tłumaczenie tego, co najważniejsze** — dane są środkiem, nie celem:

> Nie chodzi o trenowanie artefaktu dla samego trenowania (maksymalne dopasowanie do
> danych to droga do przeuczenia). Trening jest **kierowany ku `golden_calibration`** —
> szukamy skalibrowanego, NIE-przeuczonego opisu rynku (zakresy wartości cech w XGB /
> sekwencje stanów w LSTM), stabilnego na tyle, by działać jako **filtr transakcyjny ENTRY**
> przy zamrożonym kontrakcie TP/SL. To „cherry-picking of not-overfitted features": one-SE
> plateau per rodzina → najprostszy reprezentant → akceptacja ekonomiczna → propozycje
> brzegowe sterowane dowodem. Miarą sukcesu jest **audytowalny system**, który pokazuje,
> kiedy model wie wystarczająco dużo, by wskazać ENTRY — i kiedy ma pozostać bezczynny.
> Jakość i uczciwość DANYCH (w tym korekta splitów) jest fundamentem tej kalibracji:
> dobrze opisane dane → wiarygodny filtr per asset.
