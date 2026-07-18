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

## 2. Audyt przetwarzania danych — ZAKOŃCZONY (2026-07-18, 4 audytorów Opus)

Adwersaryjny audyt z POMIARAMI na realnych zapieczętowanych barach (AAPL, NVDA, ABNB,
COIN, CEG, GEHC, MSFT, VLTO + skany całego universum 498 tickerów). Wynik zbiorczy:

| Wymiar | Werdykt | Najważniejsze |
|---|---|---|
| 1. Surowe bary / struktura gapów | ⚠️ 1 BUG (major) + reszta CORRECT | luka pokrycia QC: brak kontroli kompletności sesji / płaskich barów |
| 2. Agregacja 1h→1d/1w + przyczynowość | ✅ CORRECT (0 bugów) | świece bit-w-bit; przyczynowość udowodniona trunkacją |
| 3. Etykiety / symulacja przez gapy | ⛔ 1 BUG (KRYTYCZNY) + 2 CAVEAT | dane NIEDOSTOSOWANE do splitów; benchmark HODL skażony |
| 4. NaN / warmup / eligibility / okna LSTM | ✅ CORRECT (0 bugów) | zero NaN w rdzeniu; zero imputacji; odpad okien tylko warmupowy |

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

### ⛔ ZNALEZISKO KRYTYCZNE — dane niedostosowane do splitów (potwierdzone niezależnie)
`CORP_ACTIONS_POLICY='deferred'` jest w `config/xgb.json` **walidowana ale nigdy nie
zastosowana** — bary są split-**unadjusted**. Skutki (zweryfikowane własnym pomiarem na
committed `data/results.db` + surowych barach):
- **NVDA**: surowy gap overnight −75,2% (split 4:1, 2021-07-20, w TRAIN) i −90,1%
  (split 10:1, 2024-06-10, w OOS). Trzymana pozycja przez split księguje ~−90% jako
  „stratę", której nie było.
- **Benchmark HODL skażony**: NVDA `hodl_return_pct=-56,34%`, CMG **−98,56%**, AVGO −62%,
  WMT −24% — to nie są realne zwroty buy-and-hold (prawdziwe były dodatnie/silnie dodatnie).
- **37 tickerów OOS** ma gapy splitowe >30%. Audytor: **72% wierszy skażonych splitem
  fałszywie „beats HODL"** vs 23% bazowej stopy → statystyki „beats-HODL" i część zwrotów
  ML są zawyżone/niewiarygodne dla tych tickerów.
- **Etykiety treningowe** w pobliżu splitów to artefakty (cały split złożony w jeden
  „zwrot bara").
- **Nieudokumentowane** w METHODOLOGY.md.
- **Kontrapunkt (ważny dla skali problemu)**: teza projektu jest o METODZIE i FILTRZE,
  a jednorazowy werdykt OOS jest już uczciwie NEGATYWNY („nie pobił baseline"). Split
  psuje przede wszystkim **porównanie z HODL** i pojedyncze zwroty tickerów trzymanych
  przez split — nie unieważnia mechaniki filtra ENTRY ani warstwy interpretacyjnej
  (opartej na TRAIN, oknach bez splitu w większości). Ale **liczby „beats-HODL" i
  konkretne `hodl_return_pct`/`return_pct` tickerów splitowych NIE mogą być prezentowane
  jako wiarygodne bez adnotacji lub filtra.**

### ⚠️ ZNALEZISKA WAŻNE (CAVEAT — udokumentować, nie ukrywać)
- **Detekcja barier tylko po CLOSE (BARRIER_MODE='close')** jest netto OPTYMISTYCZNA
  ~5 pp win-rate: ciaśniejszy SL (1×ATR) jest przebijany intra-bar częściej (~8,5%) niż
  TP (2×ATR, ~4,7%); w modelu first-touch win-rate spada 4,8–6,1 pp (AAPL 39,3→34,5%),
  ~12% etykiet się odwraca. Mechanizm udokumentowany, ale KIERUNEK i SKALA biasu nie.
- **Luka pokrycia QC (major)**: ingest sprawdza tylko sanity per-bar, brak kontroli
  kompletności sesji / płaskich barów. 3 ogólnorynkowe dni awarii danych zwijają całą
  sesję do JEDNEGO płaskiego bara (2021-04-19: 345/498 tickerów, 2021-10-25: 316,
  2022-03-08: 282; + 2018-05-02/03) — **3825 płaskich barów u 475 tickerów**, wszystkie
  w TRAIN, przechodzą po cichu. Na AAPL 2 płaskie bary wchodzą jako kandydaci treningowi
  i skażają 39 pobliskich wierszy (okno kroczące). Wpływ ograniczony (0,4% barów), ale realny.

### Drobne (MINOR)
- METHODOLOGY.md §2 mówi „Y=1 gdy TP przed SL", a kod etykietuje po ZNAKU zwrotu netto
  (z kosztami i gapem) — kod jest UCZCIWSZY niż dokument; poprawić zdanie w dokumencie.
- Cechy `*_alignment_multi` (nanmean) we wczesnej historii degradują do dostępnych
  interwałów zamiast zostać NaN — przyczynowe i bez wycieku, ale semantyka dryfuje;
  do udokumentowania jako zamierzone.
- Check „non-monotonic" to martwy kod (SQL ORDER BY sortuje przed testem); ścieżka
  read-from-parquet nie ma QC (ufa pieczęci) — nieszkodliwe w tym snapshotcie.

### Rekomendacja (bez naruszania FREEZE)
1. **Prezentacja**: dodać jawny disclaimer o splitach na stronie Overview/Model Comparison
   i **oznaczać/filtrować 37 tickerów splitowych** przy „beats-HODL" (albo pokazywać
   HODL tylko dla tickerów bez gapu splitowego). Nie chować.
2. **Naprawa u źródła** (dostosowanie do splitów) = ponowna budowa barów → nowa epoka
   (v5) → re-seal → rebuild — POZA obecnym zamrożonym zakresem; decyzja użytkownika.
3. Bias close-scan i lukę QC dopisać do METHODOLOGY.md jako znane ograniczenia.

## 3. Znane, udokumentowane uproszczenia (nie bugi)

- Sigma XGB = standaryzacja OPISOWA rozkładu Train (drzewa nie normalizują wejścia);
  sigma LSTM = faktyczna transformacja wejścia (NORM_STATS). Nigdy nie prezentować
  jako tego samego mechanizmu.
- Reprodukowalność results.db jest row-level (metadane builda: created_at/git_sha
  poza kontraktem bajtowym).
- Kwantyle/raty z małym n niosą flagę low_evidence (próg 10 zdarzeń ENTRY) — UI ma
  obowiązek ją pokazywać.
