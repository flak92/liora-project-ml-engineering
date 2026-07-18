# Stan weryfikacji projektu — fakty z raportów audytowych

Stan na 2026-07-18. Ten plik zbiera FAKTY z przeprowadzonych kampanii weryfikacyjnych
(raporty wieloagentowe + bramki automatyczne), żeby przy prezentacji nie trzeba było
niczego rekonstruować z pamięci. Każdy fakt ma źródło (bramka/commit/pomiar).

## 1. Co jest UDOWODNIONE (zakończone raporty)

### Warstwa modelowa i ekstrakcja interpretacji (epoka 2026-07-golden-v4)
- **Parytet scoringu**: ekstraktor interpretacji odtwarza predykcje dokładnie ścieżką
  produkcyjną pipeline'u — porównanie z `score_setups` na tickerze A: **9470/9470
  zdarzeń, max |Δp| = 0.0 (bit-exact)**.
- **Przyczynowość cech**: wartości WSZYSTKICH kolumn cech na wierszach sprzed OOS są
  **bitowo identyczne** po odcięciu barów OOS z wejścia (test trunkacji: XGB 66 kolumn,
  LSTM 59 kolumn) — kontekst 1d/1w rzutowany wyłącznie z okresów ukończonych
  (merge_asof backward po zamknięciu dnia/tygodnia).
- **Nietykalność zamrożonych store'ów**: sha256 `oos_metrics.db` i `oos_read_ledger.jsonl`
  (oba modele) **identyczne przed i po** pełnej ekstrakcji 993 tickerów; ledger
  jednorazowego odczytu OOS bez żadnego nowego wpisu.
- **Determinizm**: payload_sha256 identyczny w świeżych procesach (`--check`), a na
  czystym klonie repo re-ekstrakcja LSTM z tracked artefaktów dała **ten sam hash**.
- **Etykiety w oknie Train**: horyzont ostatniego kandydata treningowego kończy się
  przed OOS (purge `t0+H+embargo ≤ oos_start`, asercja w każdym przebiegu); detektor
  naruszenia UDOWODNIONY jako działający (zatruty timestamp → odmowa; selftest SLOW).
- **Baza prezentacyjna**: 16/16 checków integralności fail-closed, 0 WARN / 0 FAIL;
  993/993 artefaktów z manifestem SHA-256; suma udziałów kontrybucji = 1 per asset;
  klucz metodologii `interpretation_recipe_hash` jednolity w całej epoce.
- **Czysty klon**: pełny zestaw bramek zielony bez folderów roboczych Assets/
  (verify-results-db 16/0/0, testy aplikacji ALL PASS, selftest 28/28).

### Semantyka wyników (obowiązujące słownictwo)
- Model podejmuje **wyłącznie decyzję ENTRY przy ZAMROŻONYM kontrakcie TP/SL**
  (mechaniczny asymetryczny triple-barrier ATR). TP i SL nie są decyzją modelu.
- Cała warstwa interpretacyjna jest **TRAIN-DERIVED / NOT AN OOS RESULT /
  NOT A LIVE TRADING SIGNAL**; jednorazowy odczyt OOS całości jest uczciwie negatywny.
- Interwały XGB = „model-derived conditional ENTRY regions" — projekcja zachowania
  CAŁEGO modelu na jedną cechę; zapis bez cenzury (segmenty zimne też w bazie).
- 1w w plikach assetów to NIE trzecie źródło danych: agregat wyliczany z barów 1h
  (CONTEXT_TIMEFRAMES), rzutowany przyczynowo; źródła surowe = 1h (XGB) i 1d (LSTM).

## 2. Weryfikacja przetwarzania danych — ZAKOŃCZONA (2026-07-18, 4 recenzentów Opus)

Dogłębna weryfikacja z POMIARAMI na realnych zapieczętowanych barach (AAPL, NVDA, ABNB,
COIN, CEG, GEHC, MSFT, VLTO + skany całego universum 498 tickerów). Wynik zbiorczy —
rdzeń przetwarzania potwierdzony jako zdrowy; jeden obszar do domknięcia w kolejnej epoce:

| Wymiar | Wynik | Najważniejsze |
|---|---|---|
| 1. Surowe bary / struktura gapów | ✅ zdrowe + 1 usprawnienie | rozszerzyć QC o kompletność sesji / płaskie bary |
| 2. Agregacja 1h→1d/1w + przyczynowość | ✅ w pełni poprawne | świece bit-w-bit; przyczynowość udowodniona trunkacją |
| 3. Etykiety / symulacja przez gapy | ✅ mechanika poprawna + 1 domknięcie danych | dostosowanie do splitów zaplanowane na epokę v5 |
| 4. NaN / warmup / eligibility / okna LSTM | ✅ w pełni poprawne | zero NaN w rdzeniu; zero imputacji; odpad okien tylko warmupowy |

### Co jest UDOWODNIONE jako poprawne (transformacje liczbowo zdrowe)
- **Agregacja 1h→1d/1w bit-exact**: ręczna rekonstrukcja świec (open=pierwszy, high=max,
  low=min, close=ostatni, volume=suma; grupowanie po sesji ET / tygodniu ISO) = **0.0
  max błędu** vs kod i vs zapieczętowane parquety, dla 5 tickerów (w tym IPO ABNB/COIN/GEHC),
  z poprawną obsługą sesji częściowych i skróconych świąt.
- **Przyczynowość kontekstu 1d/1w**: dowód przez trunkację — obcięcie na granicy ukończonego
  tygodnia zostawia WSZYSTKIE kolumny `_1d/_1w`/multi_tf bitowo identyczne; środa w środku
  tygodnia bierze poprzedni okres; ukończony okres wchodzi dopiero na barze zamykającym
  (as-of ≤ decyzja; wejście i tak fillowane na następnym otwarciu). Zero wycieku w przód.
- **Bramki QC per-bar działają**: preparowane złe ramki (duplikat, high<low, cena≤0,
  ujemny wolumen) **rzucają RuntimeError**; skan 498 parquetów = 0 duplikatów / 0
  niemonotonicznych / 0 naruszeń OHLC / 0 ujemnego wolumenu.
- **NaN / warmup / eligibility**: liczby wiodących NaN cech kroczących = dokładnie rozmiar
  okna; `core_feature_eligibility` wyklucza KAŻDY wiersz z niefinitną cechą CORE (rdzeń df_b
  = **0 NaN / 0 Inf**), IPO-era warmup nie przecieka do kandydatów (VLTO: 49 barów warmup
  odfiltrowanych); **zero imputacji w całym kodzie** (brak fillna/interpolate/ffill/dropna —
  jedyne replace/nan_to_num to guardy dzielenia); okna LSTM odpadające na bramce skończoności
  są WYŁĄCZNIE warmupowe (ABNB/COIN: wszystkie t0≤107, zero dziur w środku historii);
  księgowość `n + n_nan == kandydaci` w payloadach zgadza się co do wiersza (pełna
  rekonsyliacja pipeline↔zapieczętowany artefakt).
- **Mechanika transakcji przez gapy uczciwa**: decyzja na close[t0], fill na PRAWDZIWYM
  otwarciu następnej sesji ze slippage (gap nocny/weekendowy poniesiony, nie fantazyjna
  cena); exit ponosi gap trigger→otwarcie; purge/embargo `t0+H+embargo ≤ oos_start`
  zweryfikowane (horyzont ostatniej etykiety Train kończy się przed OOS); wagi unikalności
  w (0,1]; TIME_BARRIER vs OOS_END_FORCED_EXIT poprawne.

### ★ OBSZAR DO DOMKNIĘCIA — dostosowanie do splitów (priorytet na epokę v5)
Bary są w wersji nieskorygowanej o splity (`CORP_ACTIONS_POLICY='deferred'` jest
zwalidowana w kodzie, samo dostosowanie zaplanowane jako osobny krok). Potwierdzone
własnym pomiarem na committed `data/results.db` + surowych barach — fakty i skala:
- **Zasięg**: **79/498 tickerów (16%)** ma gap splitowy w historii — **31 w OOS, 58 w Train**.
- **NVDA**: surowy gap overnight −75,2% (split 4:1, 2021-07-20, Train) i −90,1%
  (split 10:1, 2024-06-10, OOS) — to zmiany czysto techniczne (podział akcji), nie ruch ceny.
- **Benchmark HODL liczony na cenach surowych** pokazuje dla tych tickerów wartości
  nieskorygowane: NVDA `hodl_return_pct=-56,34%`, CMG −98,56%, AVGO −62%, WMT −24%.
  Po dostosowaniu do splitów odpowiadają one realnym, w większości dodatnim zwrotom
  buy-and-hold. Dostosowanie sprowadza je do prawdziwego HODL.
- **Statystyki „beats-HODL"** dla tych tickerów są liczone względem nieskorygowanego HODL
  i wymagają dostosowania, zanim trafią do prezentacji jako miara przewagi.
- **Etykiety treningowe** w pobliżu splitów odzwierciedlają nieskorygowany gap (cały
  podział złożony w jeden „zwrot bara") — dotyczy wąskiego okna wokół dat splitów.
- **Zakres wpływu (dobra wiadomość)**: teza projektu dotyczy METODY i FILTRA, a jednorazowy
  werdykt OOS jest już z założenia ostrożny. Dostosowanie do splitów poprawia głównie
  **porównanie z HODL** i pojedyncze zwroty 79 tickerów trzymanych przez split; **nie
  dotyka** mechaniki filtra ENTRY ani warstwy interpretacyjnej (oparte na TRAIN, w
  większości okna poza splitem). Pozostałe 419 tickerów oraz cała mechanika transakcji
  są potwierdzone jako poprawne.

### ★ DO UDOKUMENTOWANIA (drobne domknięcia jakości)
- **Detekcja barier po CLOSE (BARRIER_MODE='close')** jest z natury ostrożna po stronie
  win-rate o ~5 pp: ciaśniejszy SL (1×ATR) bywa dotykany intra-bar częściej (~8,5%) niż
  TP (2×ATR, ~4,7%); w modelu first-touch win-rate ~AAPL 39,3→34,5%. Mechanizm jest w
  kodzie i SOT; warto dopisać do METHODOLOGY.md kierunek i skalę efektu.
- **Rozszerzenie QC (usprawnienie)**: obecne bramki sprawdzają sanity per-bar; warto
  dodać kontrolę kompletności sesji / płaskich barów. 3 ogólnorynkowe dni z niepełnymi
  danymi (2021-04-19, 2021-10-25, 2022-03-08; + 2018-05-02/03) zwijają sesję do jednego
  płaskiego bara — **3825 płaskich barów u 475 tickerów (0,4% wszystkich)**, wszystkie
  w TRAIN. Wpływ niewielki i ograniczony do wąskiego okna okna kroczącego (AAPL: 39
  wierszy), ale wart jawnej bramki w v5.

### Drobne (do jednozdaniowej poprawki)
- METHODOLOGY.md §2 mówi „Y=1 gdy TP przed SL"; kod etykietuje po ZNAKU zwrotu netto
  (z kosztami i gapem) — kod jest DOKŁADNIEJSZY niż dokument; dostroić zdanie w dokumencie.
- Cechy `*_alignment_multi` (nanmean) we wczesnej historii degradują do dostępnych
  interwałów zamiast zostać NaN — przyczynowe i bez wycieku; do udokumentowania jako zamierzone.
- Check „non-monotonic" jest zabezpieczony przez SQL ORDER BY (sortuje przed testem);
  ścieżka read-from-parquet ufa pieczęci — oba nieszkodliwe w tym snapshotcie.

### Rekomendacja (bez naruszania FREEZE)
1. **Prezentacja teraz**: dodać krótką notę o splitach na stronie Overview/Model Comparison
   i **oznaczać 79 tickerów splitowych** przy „beats-HODL" (albo pokazywać HODL tylko dla
   tickerów bez gapu splitowego). Ujmować to jako świadome, jawne ograniczenie snapshotu.
2. **Domknięcie u źródła** (dostosowanie do splitów) = ponowna budowa barów → nowa epoka
   v5 → re-seal → rebuild. Szacowany koszt obliczeniowy: patrz §4.
3. Efekt close-scan i rozszerzenie QC dopisać do METHODOLOGY.md jako znane charakterystyki.

## 3. Znane, udokumentowane uproszczenia (nie bugi)

- Sigma XGB = standaryzacja OPISOWA rozkładu Train (drzewa nie normalizują wejścia);
  sigma LSTM = faktyczna transformacja wejścia (NORM_STATS). Nigdy nie prezentować
  jako tego samego mechanizmu.
- Reprodukowalność results.db jest row-level (metadane builda: created_at/git_sha
  poza kontraktem bajtowym).
- Kwantyle/raty z małym n niosą flagę low_evidence (próg 10 zdarzeń ENTRY) — UI ma
  obowiązek ją pokazywać.

## 4. Koszt obliczeniowy domknięcia splitów (wycena czasowa)

Sprzęt referencyjny: **16 rdzeni / 30 GiB** (ten serwer). Bazą wyceny są RZECZYWISTE
czasy faz nocy v4 (`night_20260718T093509Z_54cb92d`) zmierzone w tej kampanii.

**Dlaczego trzeba re-run, nie tylko poprawić liczby:** dostosowanie do splitów zmienia
ceny → zmienia cechy (returns, ATR, z-score) → zmienia etykiety triple-barrier → zmienia
modele i kalibrację. Sama korekta HODL to inna, znacznie tańsza opcja (poziom 1 niżej).

Trzy poziomy, od najtańszego:

| Poziom | Co obejmuje | Czas @16c | Uwaga |
|---|---|---|---|
| **1. Tylko HODL (read-side)** | Przeliczyć benchmark buy-and-hold na cenach skorygowanych o splity; poprawić `beats_hodl`, `hodl_return_pct` w warstwie prezentacji | **~5–15 min** | ZERO re-treningu; naprawia najbardziej mylące liczby; modele zostają jak są |
| **2. Tylko 79 tickerów splitowych** | Skorygować bary tych 79, re-seal + re-ekstrakcja tylko ich, przeliczyć HODL uniwersum | **~45–70 min** | mieszana epoka (79 nowych + 419 starych) — wymaga jawnej adnotacji spójności |
| **3. Pełna epoka v5 (rekomendowane docelowo)** | Korekta barów całego universum → pełny re-seal → żniwa → warstwa interpretacji → rebuild results.db → bramki | **~6–7 h (jedna noc)** | czysta, spójna epoka; ~100 rdzeniogodzin |

**Rozbicie poziomu 3 (z realnych czasów v4):**
- korekta barów pod splity (wektorowo na 498×~18 tys. barów 1h): **~5–15 min** (I/O-bound),
- `xgb_search` (golden-v2, 498): **~24 min** (v4: 1457 s),
- `xgb_seal` (parytet θ, 498): **~38 min** (v4: 2297 s),
- gates (dashboard+HODL): **~1 min**,
- okno LSTM (search+seal, warm-start z uniwersalnego backbone): **~2,5 h** (v4: okno 360 min, faktycznie ~150 min),
- warstwa interpretacji (ta z tej kampanii): **~40 min XGB + ~60–75 min LSTM** @12 jobów,
- rebuild results.db + bramki + testy: **~15–30 min**.
- **Suma wall-clock: ~6–7 h na 16 rdzeniach** (komfortowo jedna noc), tj. **~100 rdzeniogodzin**.

**Rekomendacja kosztowa:** na prezentację TERAZ wystarczy **poziom 1** (~kwadrans, bez
re-treningu) + nota o splitach — naprawia „beats-HODL" i `hodl_return_pct`. Pełne
domknięcie (**poziom 3**, jedna noc) zostawić jako świadomą decyzję o epoce v5. Poziom 2
zwykle nie warty zachodu (mieszana epoka komplikuje spójność za niewielką oszczędność czasu).
