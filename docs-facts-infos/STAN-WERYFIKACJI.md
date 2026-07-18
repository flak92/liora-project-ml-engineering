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

## 2. Audyt przetwarzania danych — W TOKU (agenci na Opusie)

Uruchomiony adwersaryjny audyt poprawności przetwarzania danych: 4 niezależnych
audytorów (model: Opus), każdy z obowiązkiem POMIARÓW na realnych zapieczętowanych
barach (min. 3–5 tickerów, w tym z krótką historią), werdykty CORRECT / CAVEAT / BUG:

1. **Surowe bary i struktura gapów** — bramki QC przy ingestach (monotoniczność,
   duplikaty, sanity OHLC), realna mapa braków godzin/sesji, semantyka okien
   kroczących (bar-index, nie kalendarz) wokół dziur, cechy sesyjne przy brakujących
   godzinach, tickery IPO-era (warmup × NaN).
2. **Agregacja 1h→1d/1w i przyczynowość** — ręczna rekonstrukcja świec dziennych
   i tygodniowych vs kod, granice tygodni (skrócone/pierwszy/ostatni), dostępność
   kontekstu dopiero po ukończeniu okresu, propagacja NaN wczesnej historii.
3. **Etykiety i symulacja transakcji przez gapy** — fill na otwarciu następnej sesji
   (gap nocny/weekendowy uczciwie poniesiony), detekcja barier po CLOSE (pomiar, jak
   często high/low przebija barierę, a close nie — klasyfikacja: konserwatywne
   uproszczenie czy nieudokumentowany bias), ścieżki GAP_INVALIDATED_SKIP /
   INVALID_BARRIER_SKIP, TIME_BARRIER vs koniec danych, wagi unikalności.
4. **NaN / warmup / eligibility / bramka okien LSTM** — zero NaN w rdzeniu macierzy
   treningowej (eligibility), polityka missing-branch dla cech opcjonalnych XGB
   (zero imputacji w całym kodzie), ile okien LSTM odpada na bramce skończoności
   i CZY odpad jest tylko warmupowy (nie dziury w środku historii), guardy dzielenia
   na płaskich/zerowych barach, spójność księgowości n_nan w payloadach.

**Wyniki audytu zostaną dopisane do tego pliku po zakończeniu przebiegu.**

## 3. Znane, udokumentowane uproszczenia (nie bugi)

- Sigma XGB = standaryzacja OPISOWA rozkładu Train (drzewa nie normalizują wejścia);
  sigma LSTM = faktyczna transformacja wejścia (NORM_STATS). Nigdy nie prezentować
  jako tego samego mechanizmu.
- Reprodukowalność results.db jest row-level (metadane builda: created_at/git_sha
  poza kontraktem bajtowym).
- Kwantyle/raty z małym n niosą flagę low_evidence (próg 10 zdarzeń ENTRY) — UI ma
  obowiązek ją pokazywać.
