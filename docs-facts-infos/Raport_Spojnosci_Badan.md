# Raport spójności i poprawności prac badawczych nad algorytmem decyzji transakcyjnych w przestrzeniach giełdowych bazujących na komunikacji wartościami OHLCV opartym o machine- i deep-learning

**Przedmiot audytu:** repozytorium `liora-project-ml-engineering`, epoka `2026-07-golden-v5`.
Audyt wykonano na gałęzi badawczej `to_give_up_and_show`; w tej kopii dokumentu referencje
plików przepisano na układ gałęzi prezentacyjnej `Stable_Presentable_Version`, a odwołania do
warstwy orkiestracji, która nie jest częścią tej gałęzi, oznaczono „(gałąź badawcza)".
**Stan badania:** `research_status = FROZEN_FINAL_RESEARCH_SNAPSHOT`, zamrożenie prezentacyjne
`presentation-v1/2026-07-golden-v5`.
**Data audytu:** 2026-07-19.

Wszystkie liczby w tym dokumencie zostały **zmierzone** — zapytaniem do `data/results.db`,
parsowaniem ledgerów odczytów OOS (gałąź badawcza; ich skumulowane liczniki są zsumowane
w tabeli `oos_read_summary` wewnątrz `data/results.db` i pokazane na stronie Integrity) albo
odczytem wskazanego miejsca w kodzie. Żadna nie została przepisana z prozy innego dokumentu.
Tam, gdzie liczba pochodzi z pliku, podano `plik:linia`.

---

## 1. Werdykt

**Badanie można przedstawić jako poprawne metodologicznie — pod zadeklarowanymi ograniczeniami
z sekcji 4, które są częścią wyniku, a nie przypisem do niego.**

Rdzeń jest zdrowy w sposób, który daje się sprawdzić, a nie tylko zadeklarować: okno
out-of-sample nie zasila żadnej decyzji po stronie Train, każdy jego odczyt jest policzony
w dopisywalnym ledgerze, punkt pracy wybierany jest jednym wspólnym kryterium zamiast najlepszego
per fold, a warstwa interpretacji zapisuje również segmenty niekorzystne, zamiast pokazywać
wyłącznie te potwierdzające tezę.

Najważniejsza cecha tego projektu z perspektywy audytu jest taka, że **główny wynik jest negatywny
i tak też jest raportowany**. Mediana strategii przegrywa z prostym trzymaniem waloru w obu
rodzinach modeli. Projekt nie próbuje tego zamaskować doborem metryki ani zakresu — i to jest
mocniejszy argument za poprawnością warsztatu niż jakikolwiek wynik dodatni.

Czego ten dokument **nie** stwierdza: że opisany system ma przewagę rynkową, że nadaje się do
użycia na żywo, ani że wynik uogólnia się poza zbadane okno.

---

## 2. Tożsamość badania

| Pole | Wartość | Źródło |
|---|---|---|
| Epoka | `2026-07-golden-v5` | `research_run` |
| Recipe hash XGB | `447745d1059e560f` | `research_run` |
| Recipe hash LSTM | `a4cc8f4a78ad8574` | `research_run`, przeliczalny z modułu ledgera OOS (gałąź badawcza) |
| Okno Train / OOS — XGB | `2016-10-17 → 2023-12-29` / `2024-01-02 → 2026-05-29` | `research_run` |
| Okno Train / OOS — LSTM | `2017-01-01 → 2023-12-31` / `2024-01-01 → 2026-04-30` | `research_run` |
| Zapieczętowane wiersze | 498 XGB + 495 LSTM | `asset_results` |
| Artefakty | 993 katalogi, po 5 plików | `artifacts/{xgb,lstm}/`, manifest `_meta.counts` |
| Kontrole integralności | **16 / 16 PASS** | `integrity_checks` |

Uwaga o tożsamości na tej gałęzi: pola identyfikacyjne `research_run` w `data/results.db` są
zanonimizowane (`epoch = 'sealed'`, `run_id = 'sealed-final'`,
`presentation_freeze = 'public/stable-2'`, `git_sha = NULL`). Wartości epoki i zamrożenia
w tabeli powyżej pochodzą z audytu wykonanego na gałęzi badawczej; recipe hashe i okna czasowe
są w obu miejscach identyczne.

`recipe_hash` obejmuje **konfigurację metody**, nie dane wejściowe. Nie jest to przeoczenie, lecz
świadomy rozdział: tożsamość barów jest zapisana osobno (polityka corporate actions
i `events_sha256` w `oos_read_summary.reason`), dzięki czemu korekta danych może zostać powtórzona
i porównana wobec **tego samego** hasha metody. Złożenie obu w jeden skrót sprawiłoby, że każda
poprawka danych wyglądałaby jak zmiana metody. Rozdział jest zadeklarowany wprost w
`src/xgb/feature_search.py` w dokumentacji funkcji `recipe_hash` (`:93`).

---

## 3. Co zostało sprawdzone i uznane za solidne

### 3.1 Izolacja okna out-of-sample

Kontrola dotyczyła pytania: czy istnieje **jakakolwiek** ścieżka, którą wartość z OOS wpływa na
trening, dobór cech, HPO lub punkt pracy.

- Przycięcie zbioru Train jest twardą asercją, nie filtrem najlepszych intencji:
  `src/xgb/pipeline.py:322-323` zachowuje wyłącznie zdarzenia spełniające
  `t0 + H + embargo <= oos_start`, po czym **asertuje**, że żadne okno etykiety nie sięga OOS.
- Ta sama granica jest egzekwowana wewnątrz walidacji krzyżowej (`src/xgb/pipeline.py:848`,
  `:933`) — każde okno silnika w każdym foldzie musi kończyć się przed początkiem OOS.
- Odczyt OOS ma jeden **choke point** z bramką w runnerze badawczym (gałąź badawcza; runner nie
  jest częścią tej gałęzi, por. `docs/ARCHITECTURE.md`): odmawia odczytu, gdy w ledgerze nie ma
  otwartej epoki. Zamiast ostrzeżenia następuje przerwanie.

**Wniosek:** nie znaleziono ścieżki przecieku z OOS do decyzji treningowych.

### 3.2 Dyscyplina odczytów — policzona, nie założona

To jest miejsce, w którym projekt mówi o sobie rzecz niewygodną, i robi to poprawnie.

| Pipeline | Cykli epoki | Odczytów | Tickerów | Odczytanych > 1 raz |
|---|---|---|---|---|
| XGB | 3 | **588** | 498 | **89** |
| LSTM | 2 | **495** | 495 | **0** |

Rozbicie XGB na cykle open/close: cykl 2 to **przerwana próba** (89 odczytów / 89 tickerów, zero
powtórzeń wewnątrz cyklu), cykl 3 to właściwe pieczętowanie (499 odczytów / 498 tickerów,
powtórzony wyłącznie AAPL). Skumulowany licznik odczytów na ticker wynosi 4–9 (średnia 5,18) dla
XGB i 4–6 (średnia 4,00) dla LSTM.

Twierdzenie „każde aktywo odczytano dokładnie raz" byłoby zatem **nieprawdziwe** i zostało
z projektu usunięte. Obowiązuje sformułowanie odpowiadające danym: *okno OOS nie jest wejściem do
żadnej decyzji po stronie Train, a każdy jego odczyt jest zapisany w dopisywalnym ledgerze; liczba
odczytów w epoce bywa większa od liczby aktywów i jest raportowana wprost.*

Uzasadnienie tej konstrukcji jest zapisane w samym ledgerze (moduł ledgera OOS, gałąź badawcza):
ponowne pieczętowanie jest dozwolone, natomiast **nieodnotowany** ponowny odczyt byłby nie do
odróżnienia od przebierania w wynikach.

### 3.3 Kontrakt etykiety i egzekucji

Etykieta **nie** oznacza „TP osiągnięto przed SL". Oznacza dodatni wynik netto po przejściu całego
kontraktu wykonania — `src/lstm/pipeline.py:354`:

```
y = 1 if sim["local_per_unit_net_return"] > 0 else 0
```

Na ten wynik składają się: wyzwolenie bariery **na zamknięciu** (`BARRIER_MODE: close`), wejście po
otwarciu następnego słupka (`ENTRY_FILL: next_bar_open`), wyjście po otwarciu słupka następującego
po wyzwoleniu (`EXIT_FILL: trigger_next_open`), wyjście czasowe na zamknięciu
(`SCHEDULED_EXIT_FILL: scheduled_moc_close`), prowizja 1 bp i poślizg 2 bp po obu stronach oraz
bariera czasowa `H`. Zdarzenia z luką unieważniającą lub niepoprawną geometrią bariery są
odrzucane (`GAP_INVALIDATED_SKIP`, `INVALID_BARRIER_SKIP`).

Bariery są testowane **wyłącznie wobec ceny zamknięcia** (`BARRIER_MODE: close`;
`target_hit = s*(c[t]-tp) >= 0`). Maksima i minima słupków nie są odczytywane, mimo że są
w danych. Skan po zamknięciu przeoczy dotknięcia śróddzienne — a że stop leży bliżej (1×ATR) niż
cel (2×ATR), przeocza częściej stopy. Efekt jest zatem **zachowawczy wobec win-rate**, nie
optymistyczny, ale jest realnym uproszczeniem kontraktu i tak go tu nazywamy.

Konsekwencja, którą projekt teraz nazywa wprost: **nominalna geometria 2:1 nie jest zrealizowanym
payoffem.** Zmierzony stosunek średniej wygranej do średniej przegranej (wyliczony z `profit_factor`
oraz liczby wygranych i przegranych) wynosi:

| Model | Mediana | p25 | p75 | Udział aktywów ≥ 2,0 |
|---|---|---|---|---|
| XGB | **1,447** (n = 328) | 1,340 | 1,535 | **1,2 %** |
| LSTM | **1,241** (n = 435) | 1,033 | 1,500 | **7,6 %** |

Zdanie „jedna wygrana pokrywa dwie przegrane" byłoby więc prawdziwe wyłącznie dla wyidealizowanego,
bezkosztowego payoffu `+2R/−1R`. Zostało z projektu usunięte.

### 3.4 Punkt pracy i pojęcie strategii promowanej

`src/shared/op_select.py` gromadzi wyniki **po wszystkich foldach** i wybiera **jeden wspólny** punkt
pracy dla całego okna Train. Najlepsza theta per fold jest jawnie zakazana jako „fold oracle" —
uzasadnienie w dokumentacji modułu: żadna wdrażalna strategia nie może przełączać progu między
foldami. Kolejność kryteriów: próg transakcji (`min_oof_trades`), następnie rozrzut między foldami.

Gdy nic nie przechodzi progu, punkt jest oznaczany `trade_floor_met = False`, a moduł stwierdza
wprost, że **wywołujący nie może promować takiego wyniku**.

Rozkład wybranych theta pokazuje, że kalibracja realnie pracuje, a nie zwraca stałej:

- XGB (siatka 0,40–0,60): `0,40 → 199`, `0,44 → 79`, `0,60 → 140`, reszta rozproszona;
- LSTM (siatka 0,50–0,60): `0,50 → 178`, `0,52 → 100`, `0,54 → 98`, `0,56 → 63`, `0,58 → 35`, `0,60 → 21`.

**Rozróżnienie, które musiało zostać doprecyzowane.** Stan `TRAIN_OOF_FLOOR_NOT_MET` nie jest
tożsamy z bezczynnością modelu. W całym uniwersum istnieje **dokładnie jeden** wiersz, w którym
model handlował, a konfiguracja nie została promowana: **LSTM CEG, 40 transakcji modelu**, wynik
`+136,09 %` przy benchmarku `+168,91 %`. Wszystkie 136 wierszy XGB z niespełnionym progiem ma zero
transakcji — tam „bezczynny" jest opisem trafnym.

Dlatego agregat nazywany wynikiem strategii liczony jest przez `trade_floor_met = 1 AND
model_trades > 0`, a diagram porównawczy kwalifikuje aktywo po zapieczętowanej etykiecie
`result_mode = 'ML_MULTI_TRADE'`. Sprawdzono, że etykieta ta jest **dokładnie równoważna**
warunkowi `trade_floor_met = 1 AND model_trades >= 2` — zero rozbieżności na 993 wierszach.

### 3.4b Jak faktycznie dobrano hiperparametry LSTM

Audyt wykazał rozbieżność między opisem a przebiegiem. Aplikacja mówiła czytelnikowi, że oba
modele stroi Optuna. **Dla LSTM w tej epoce Optuna per asset nie biegła.** Gdy istnieje
zacommitowany backbone (gałąź badawcza: `lstm/data/universal_backbone.json`), runner badawczy
bierze architekturę z `warm["arch"]` **zamiast** wołać `M.hpo(...)`. Backbone niesie jedną
architekturę (`hidden 32`, `num_layers 1`, `dropout 0.3`, `lr 0.001`, `weight_decay 1e-4`),
wspólną dla wszystkich 495 aktywów; per-asset pozostaje kalibracja punktu pracy (θ, kierunek)
i wybór cech.

Opis w aplikacji został poprawiony tak, by mówił to wprost, wraz z zastrzeżeniem o czystości
foldów. Ścieżka cold-start z Optuną per asset istnieje i jest wyzwalana zmienną
`LSTM_COLD_START=1`, ale nie użyto jej do zapieczętowania tej epoki.

### 3.5 Dane wejściowe i zdarzenia korporacyjne

Korekta splitów jest nakładana na barach **godzinowych, przed jakąkolwiek agregacją**. Powód jest
arytmetyczny i podany w kodzie: skalowanie jest przemienne z `first/max/min/last`, ale **nie**
z całkowitą sumą wolumenu — korekta po agregacji zepsułaby wolumen.

Tabela detektora zawiera 88 zdarzeń na 73 tickerach; po 28 wpisach nadpisujących (usunięcia
fałszywych trafień, dodania zdarzeń niewykrywalnych z barów) obowiązujący stan to **83 zdarzenia
na 69 tickerach**, `events_sha256 = 5989535b02f384f8`, zapisany w `oos_read_summary.reason` obu
pipeline'ów.

### 3.6 Warstwa interpretacji

Warstwa czyta zapieczętowany artefakt i wiersze Train — **nigdy okna OOS**. Pokrycie: 16 601
wierszy statystyk cech, 16 601 wierszy wkładu, 45 021 segmentów zakresów ENTRY.

Najmocniejszy dowód braku doboru pod tezę: z 45 021 segmentów tylko **12 472** jest oznaczonych
flagą `candidate_entry_region`, a pozostałe **32 549 zapisano mimo braku wyróżnienia**. Flaga jest
zatem wyróżnieniem prezentacyjnym, nie filtrem zapisu — segmenty niekorzystne są w danych.

Każdy ładunek niesie w sobie trzy etykiety obowiązkowe, więc nie da się ich zgubić przy
przetwarzaniu: `TRAIN-DERIVED INTERPRETATION`, `NOT AN OOS RESULT`, `NOT A LIVE TRADING SIGNAL`.

### 3.7 Wynik — raportowany bez wygładzania

| | XGB (1h) | LSTM (dzienny) |
|---|---|---|
| Aktywa | 498 | 495 |
| Mediana zwrotu strategii | **−1,78 %** | **+2,25 %** |
| Mediana buy & hold (to samo okno) | **+22,16 %** | **+23,09 %** |
| Mediana profit factor | **0,935** (n = 328) | **1,013** (n = 445) |
| Pokonuje własny benchmark | **72 / 498 (14,5 %)** | **129 / 495 (26,1 %)** |
| Mediana transakcji modelu | 86,5 | 28 |

Podział `result_mode`: XGB — 328 promowanych, 136 bez spełnionego progu, 33 fallback do benchmarku,
1 pojedyncza transakcja; LSTM — 446 / 1 / 37 / 11.

---

## 4. Zadeklarowane ograniczenia

Poniższe nie są defektami. Są warunkami, bez których wynik zostałby odczytany szerzej, niż na to
pozwala.

1. **Survivorship.** Uniwersum to dzisiejsi członkowie indeksu zastosowani wstecz. Wyniki
   zagregowane są przez to optymistyczne.
2. **Okno OOS w hossie.** Lata 2024–2026 czynią z buy & hold benchmark wyjątkowo trudny; wynik
   nie mówi, jak metoda zachowałaby się w innym reżimie.
3. **Dywidendy nieskorygowane.** Korygowane są wyłącznie splity. Benchmark jest zatem
   **cenowy** — i to właśnie utrzymuje go na tej samej płaszczyźnie co strategia.
4. **Detektor splitów.** Proporcje poniżej 3:2 nie są wykrywalne z barów, bo mieszczą się
   w zakresie zwykłych ruchów po wynikach. Tabela przeszła przegląd człowieka, ale nie zewnętrzną
   weryfikację wobec niezależnego źródła zdarzeń korporacyjnych.
5. **Czystość foldów przy warm-starcie LSTM.** Backbone inicjalizujący modele LSTM był uczony na
   całym regionie Train, więc inicjalizacja pojedynczego foldu widziała wiersze Train spoza tego
   foldu. **Nie jest to przeciek z OOS**, ale walidacja LSTM nie jest w pełni fold-causal.
   Metodologia jest izolująca OOS; nie jest bezwarunkowo „leak-free".
6. **Kumulacja odczytów OOS.** Opisana w 3.2. Zapieczętowany wynik jest warunkowany wiedzą
   z wcześniejszych odczytów tego samego okresu.
7. **Zakres weryfikatorów.** Weryfikatory `make verify-*` działają na gałęzi badawczej
   i odtwarzają **próbkę** wierszy: XGB porównuje 5 pól na committed store'ze 15 tickerów, LSTM
   4 pola na próbce 10 tickerów, oba z tolerancją `1e-6`. Jest to deterministyczna reprodukcja
   wybranych wierszy, **nie** identyczność bajtowa i **nie** całe uniwersum — pełny store
   godzinowy XGB nie jest częścią repozytorium. Na tej gałęzi drzewo artefaktów jest weryfikowane
   skrótami SHA-256 per katalog (`artifacts/manifest.json`), a dwa wykonane notebooki w
   `examples/` pokazują odtworzone wiersze XGB.
8. **Interpretacja jest in-sample.** Opisuje zachowanie modelu na oknie Train. Nie jest wynikiem
   OOS ani sygnałem transakcyjnym.

---

## 5. Ścieżka remediacji

Audyt nie zastał projektu w stanie końcowym — poprzedziła go seria poprawek (commity gałęzi
badawczej), którą odnotowujemy, bo jest częścią ścieżki audytowej.

| Commit | Czego dotyczył |
|---|---|
| `dcde5b97` | jedna warstwa dostępu do danych; rozdzielenie wyniku modelu, ścieżki wykonanej i benchmarku; strona Integrity |
| `e4f0c8d7` | liczby z poprzedniej epoki w README; dwa fałszywe komentarze opisujące mechanizm |
| `cac93edf` | 12 korekt spójności: predykat strategii promowanej, payoff, „leak-free", zakres reprodukowalności, teza README |
| `8f61e288` | naprawa weryfikatora LSTM; zakresowanie klauzuli, którą uczyniła fałszywą wcześniejsza poprawka |
| `fd403caa` | ostatnie deklaracje o jednym odczycie i notacji zbiorów |

Dwie obserwacje z tego przebiegu warto zapisać, bo dotyczą warsztatu, a nie wyniku:

**Notebooki prezentacyjne były o epokę do tyłu — i twierdziły, że są aktualne.** Cztery
zacommitowane notebooki (dwa renderowane przez stronę aplikacji) pochodziły z epoki przed korektą
splitów i drukowały wynik NVDA `436,32 USD / −56,37 % / 1 transakcja`, opatrzony zdaniem
„REPRODUCED the sealed row within 1e-6". Zapieczętowany wiersz v5 mówi `3 390,08 USD / +239,01 % /
288 transakcji`. Twierdzenie o odtworzeniu było więc **fałszywe wobec bieżącego store'u**, i było
widoczne w prezentowanej aplikacji.

Dodatkowo okazało się, że były to nie tylko stare *wyniki*, ale i stary *kod*: komórka wołała
`strategy_meta(..., BEST, {}, ...)`, przekazując pusty rekord kalibracji tam, gdzie pipeline v5
wymaga punktu pracy z `op_select`. Notebooki XGB odtworzono, wykonując **kanoniczny szablon**
per-asset (gałąź badawcza: `xgb/src/notebook_template.ipynb` — ten sam, którego używa runner
badawczy); notebooki LSTM zregenerowano wiernie z bieżącego runnera LSTM (gałąź badawcza), wraz
ze ścieżką warm-start. Wszystkie cztery przebiegi wykonano wobec **scratch store'u**
(`OOS_METRICS_DB`), więc nie były to odczyty pieczętujące i ledgery pozostały nietknięte. Wynik:
**4 / 4 odtwarzają zapieczętowane wiersze** (XGB NVDA `3390,0761316752228`, XGB AAPL
`941,9721600160755`, LSTM NVDA `1363,2486792394723`, LSTM AAPL `874,7211883830511`). Na tej
gałęzi dwa wykonania XGB są zacommitowane jako `examples/{AAPL,NVDA}_XGB.ipynb`.

**Poprawki potrafią wprowadzać błędy.** Zacieśnienie predykatu strategii promowanej przesunęło CEG
do kubełka „nie promowane", przez co podpis tego kubełka — „ich ścieżka kapitału jest ścieżką
benchmarku" — stał się fałszywy właśnie dla CEG (2 360,89 USD wobec benchmarku 2 689,10 USD).
Błąd wykryła dopiero niezależna weryfikacja adwersaryjna, nie autor poprawki.

Drugi przykład z tej samej rodziny: rozdzielając wynik modelu od ścieżki wykonanej, użyłem słowa
„promoted" dla agregatu liczonego jako `trade_floor_met = 1 AND model_trades > 0` (786 wierszy),
podczas gdy podpis diagramu definiował „promoted" jako `ML_MULTI_TRADE`, czyli floor **i co
najmniej dwie** transakcje (774 wiersze). Predykaty są zasadne osobno — udziały wygranych
wymagają co najmniej dwóch transakcji, żeby cokolwiek znaczyły — ale jedno słowo na dwa różne
zbiory jest niespójnością. Rozdzielono słownictwo: agregat to **wynik modelu**, diagram operuje
**promowaną strategią**.

**Bramki dosłowne są kruche.** Kontrola szukająca dokładnych ciągów przepuściła wariant
sformułowania („one win **can** cover two losses") oraz ten sam zwrot zapisany inną wielkością
liter. Dalszy przegląd prowadzono już wzorcami, nieczułymi na wielkość liter. Zastrzeżenie dla
czytelnika: była to **procedura audytu**, nie artefakt repozytorium — w repo nie ma
zacommitowanej bramki skanującej tekst pod kątem zakazanych twierdzeń, więc kontrola nie
powtarza się sama i każdy kolejny przegląd trzeba przeprowadzić świadomie.

Przy tej okazji wykryto i naprawiono defekt czynny: domyślna ścieżka `make verify-lstm` (gałąź
badawcza) przerywała się wyjątkiem, odkąd plik nadpisań cech zmienił kształt w epoce v5 —
komenda, którą dokumentacja każe uruchomić czytelnikowi, nie weryfikowała niczego. Po naprawie
uruchomiono ją: **10 / 10 wierszy odtworzonych**.

---

## 6. Co zostanie zaatakowane najpierw

**„Twierdzicie, że OOS czytacie raz, a ledger pokazuje 588 odczytów na 498 aktywów."**
Nie twierdzimy tego i nigdzie tak nie napisano. Gwarancją nie jest jednokrotność odczytu, lecz to,
że **każdy odczyt jest policzony**. 588 odczytów pochodzi z przerwanego pieczętowania (89 aktywów)
i przebiegu właściwego (499 odczytów, jedno powtórzenie — AAPL). Skumulowane liczniki są pokazane
na stronie Integrity. Konsekwencję nazywamy wprost: wynik jest warunkowany wcześniejszymi odczytami
tego samego okresu.

**„Skoro bariery są 2:1, jedna wygrana pokrywa dwie przegrane — więc wystarczy 33 % trafień."**
Nie wystarczy, bo 2:1 to geometria nominalna, nie zrealizowany payoff. Zmierzona mediana stosunku
wygranej do przegranej to 1,447 (XGB) i 1,241 (LSTM); progu 2,0 sięga 1,2 % i 7,6 % aktywów.
Różnicę tworzą wyzwolenie na zamknięciu, wejście i wyjście po cenach otwarcia następnych słupków,
luki, koszty po obu stronach oraz wyjścia czasowe.

**„Czy to jest metodologia bez przecieku?"**
Bez przecieku z OOS — tak, i jest to egzekwowane asercjami, nie deklaracją. Ale nie
bezwarunkowo „leak-free": universal warm-start LSTM był uczony na całym regionie Train, więc
walidacja krzyżowa LSTM nie jest w pełni fold-causal. Nazywamy to ograniczeniem izolacji OOS
z udokumentowanym zastrzeżeniem co do czystości foldów, a nie brakiem przecieku w ogóle.

---

## 7. Podsumowanie

Deterministyczna, zamrożona prezentacja akademickiego eksperymentu porównującego XGBoost i LSTM
w selekcji transakcji na danych S&P 500. Okno out-of-sample pozostaje odseparowane od decyzji po
stronie Train, wszystkie jego odczyty są jawnie rejestrowane, a wynik finalnej epoki jest
**negatywny**. Artefakty per aktywo są weryfikowane skrótami, a wybrane wiersze wynikowe
deterministycznie reprodukowane z committed fixtures w zadeklarowanej tolerancji.

Projekt nie stanowi systemu działającego na żywo ani dowodu istnienia rentownej przewagi.
Produktem jest **metoda i jej audytowalność**: opis, w jakich warunkach model podejmuje decyzję
wejścia, kiedy świadomie pozostaje bezczynny, oraz co dokładnie zostało zmierzone i czego nie
zmierzono.
