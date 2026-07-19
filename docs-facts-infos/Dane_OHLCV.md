# Dane OHLCV — źródła, przetwarzanie i weryfikacja

Ten dokument opisuje **dane** w projekcie: skąd pochodzą, jak są przetwarzane w cechy
i etykiety, oraz co potwierdziła weryfikacja na realnych zapieczętowanych barach.
Wszystkie fakty mają źródło (plik:linia / pomiar). Stan: 2026-07-19, epoka **`2026-07-golden-v5`** (bary skorygowane o splity),
gałąź `Stable_Presentable_Version`.

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

## 4. Splity — DOMKNIĘTE (epoka 2026-07-golden-v5, 2026-07-19)

Błąd opisany wcześniej (bary nieskorygowane o splity) **został naprawiony u źródła**
i cały pipeline przeliczony na skorygowanych danych.

**Jak naprawiono.** Autorytatywnego źródła współczynników nie było na serwerze (LEAN ma
tylko 22 przykładowe tickery, bez NVDA/AMZN/AVGO; API dostawcy zwraca 401), więc tabela
zdarzeń powstała **z danych + przeglądu człowieka**:
- detektor o **zawężonym** zbiorze ratio `{3:2, 2..8, 10, 15, 20, 25, 50}` + odwrotności —
  gęsta siatka (8:5, 5:3, 7:4) sprawiała, że **krachy dopasowywały się lepiej niż prawdziwe
  splity** (krach FISV −44% trafiał w 8:5 z resztą 0,10%), co dawało 17,6% błędu w obie strony;
- **przegląd wszystkich 126 kandydatów**: 88 auto-zaakceptowanych, z tego 10 usunięto jako
  fałszywki (spinoffy DELL/DD/PNR, dywidenda specjalna KDP, krachy PG&E/OPEC/CVNA/TTD/VRT),
  a 5 dopisano ręcznie (SHW 3:1 i ROL 3:2 tuż za progiem; HLT i DD 1:3 odwrotne wplecione
  w spinoffy; EXE 1:200 zniekształcone upadłością). **Wynik: 83 zdarzenia / 69 tickerów**,
  każde z uzasadnieniem w `overrides.csv`. Sama tabela nadpisań liczy łącznie **28 wpisów**;
  ich efektem netto na zbiorze auto-zaakceptowanych są opisane usunięcia i dodania
  (88 → 83 zdarzeń). Kanoniczna księgowość: `Raport_Spojnosci_Badan.md` §3.5.
- Korekta nakładana w `xgb/src/bars.py:load_bars()` **na barach 1h, PRZED roll-upem**
  (ceny × faktor, wolumen ÷ faktor); dzienny store LSTM rolluje się z tego samego
  skorygowanego strumienia, więc 1h i 1d są spójne z konstrukcji.
- `CORP_ACTIONS_POLICY` przestała być martwą konfiguracją — `A_adjusted` realnie rozgałęzia kod.

**Dowody poprawności (zmierzone, nie deklarowane):**
| Test | Wynik |
|---|---|
| Kotwica LEAN (AAPL) | faktor **0,25 → 1,00** — trafiony dokładnie |
| Negatywna kontrola | **434 tickery bez zdarzeń bit-identyczne** z surowymi |
| **Chirurgiczność korekty** | z 497 modeli XGB porównywalnych między epokami **436 bit-identycznych z v4**, zmieniło się tylko **61** — dokładnie te ze splitami (uniwersum v5 pieczętuje 498 wierszy XGB) |
| Gapy po korekcie | NVDA −90,1% → −0,77%, CMG −98% → +0,98%, AMCR +404,6% → +0,93% |
| Spójność 1h↔1d | roll-up == store, dokładnie (2612 sesji) |
| Bramki cross-bar | strzelają na **każdym** surowym splicie, **0/503 alarmów** po korekcie |

**Skutek dla wyników** (te liczby były wcześniej nieprawdziwe):
| | v4 (surowe) | v5 (skorygowane) |
|---|---|---|
| NVDA HODL | −56,3% | **+336,6%** |
| CMG HODL | −98,6% | **−27,9%** |
| AVGO HODL | −62,4% | **+276,5%** |
| AMCR HODL | +294,0% | **−20,4%** (miraż odwrotnego splitu) |
| beats-HODL XGB | 85/498 | **72/498** |
| beats-HODL LSTM | 143/495 | **129/495** |
| tickery z HODL < −50% | 33 | **22** |

Niezależny przegląd adwersaryjny przewidział te wartości z samej tabeli zdarzeń
(NVDA ~+339%, CMG ~−25%, AVGO ~+280%, AMCR ~−21%, beats-HODL 72 i 131) — pełny retrain
je potwierdził, co jest mocnym dowodem, że korekta zrobiła dokładnie to, co miała.

**Świadome ograniczenia, zapisane wprost:**
- splity **poniżej 3:2 nie są wykrywalne z barów** (skan przy |gap|>10% daje ~1000
  przypadkowych zbieżności) — pomijamy je i to dokumentujemy;
- **spinoffy i dywidendy specjalne NIE są korygowane** — dla benchmarku cenowego to realne
  spadki, ta sama klasa co dywidenda;
- korroboracja wolumenem jest **flagą do przeglądu, nie bramką** (zawodzi w obie strony);
- jedyna zewnętrzna kotwica prawdy to plik LEAN dla AAPL.

## 5. Drobne domknięcia jakości danych (usprawnienia)

- **Rozszerzenie QC**: obecne bramki sprawdzają sanity per-bar; warto dodać kontrolę
  kompletności sesji / płaskich barów. 3 ogólnorynkowe dni z niepełnymi danymi
  (2021-04-19, 2021-10-25, 2022-03-08; + 2018-05-02/03) zwijają sesję do jednego płaskiego
  bara — **3825 płaskich barów u 475 tickerów (0,4% wszystkich)**, wszystkie w Train, wpływ
  niewielki i ograniczony do wąskiego okna kroczącego. **Domknięte:** w v5 objęte jawną
  bramką (`cross_bar_qc`, płaskie bary z progami udziałowymi).
- **Detekcja barier po CLOSE** jest z natury ostrożna po stronie win-rate o ~5 pp (ciaśniejszy
  SL 1×ATR bywa dotykany intra-bar częściej niż TP 2×ATR); mechanizm jest w kodzie.
  **Domknięte:** kierunek i skala są opisane w `docs/METHODOLOGY.md` §6 (wiersz „Barrier
  timing").
- **Dostrojenie zdania w dokumencie** — **Domknięte:** `docs/METHODOLOGY.md` definiuje już
  etykietę po znaku zwrotu netto („Y = 1 iff the realized net return … is > 0", z dopiskiem,
  że to surowszy warunek niż „TP przed SL"), zgodnie z kodem.
- Cechy `*_alignment_multi` we wczesnej historii degradują do dostępnych interwałów zamiast
  zostać NaN — przyczynowe i bez wycieku; do udokumentowania jako zamierzone.

## 6. Kolejność prac i sedno przekazu o danych

1. **Korekta splitów** — zrealizowana w epoce v5 (sekcja 4 — DOMKNIĘTE).
2. **Dalej z designem** — dziewięć stron konsoli Streamlit nad gotowym `data/results.db`
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
