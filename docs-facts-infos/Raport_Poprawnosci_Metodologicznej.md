# Raport samodzielnego sprawdzenia integralności metodologicznej systemu ML

> **Status weryfikacji (2026-07-18):** wszystkie referencje plik:linia oraz twierdzenia mechanizmowe tego raportu zostały niezależnie sprawdzone bezpośrednio w kodzie — na gałęzi badawczej `to_give_up_and_show` (commit `d8f920f`) oraz w drzewie `src/` gałęzi `Stable_Presentable_Version`. Wynik: **wszystkie mechanizmy potwierdzone w kodzie**; względem tekstu pierwotnego naniesiono cztery doprecyzowania: (1) linie `848/933` w XGB to asserty fail-closed, a replay do `hi` wykonuje `op_grid_scores` (`pipeline.py:802–803`); (2) choke pointy odczytu OOS znajdują się w entrypointach `run_asset.py`, a rzeczywiste `read_count` uniwersum to XGB = 4 / LSTM = 3 (wartości historyczne sprzed re-sealu v5; aktualne liczby — 588/495 odczytów, średnie 5,18/4,00 — w §5.2); (3) walidator DSL propozycji jest mechanizmem podsystemu LSTM (XGB wybiera z zamkniętego rejestru cech); (4) przy cytowanych dokumentach dopisano gałąź, na której istnieją.
>
> **Mapa referencji do kodu:** referencje liniowe w tabelach odnoszą się do układu gałęzi badawczej (`xgb/src/pipeline.py`, `lstm/pipeline.py`, `lstm/model.py`, `lstm/feature_search.py`, `lstm/pretrain_universal.py`, `iterators/…`). Odpowiedniki na gałęzi `Stable_Presentable_Version`: `src/xgb/pipeline.py` (linie identyczne), `src/lstm/pipeline.py` (offset +1 dla konstrukcji pipeline), `src/lstm/model.py` i `src/lstm/feature_search.py` (linie identyczne), `src/shared/op_select.py` (dawniej `iterators/op_select.py`), `src/xgb/artifact.py` (dawniej `xgb/src/asset_writers.py`), `docs/METHODOLOGY.md` (odpowiednik `docs/FEATURE_SEARCH_METHODOLOGY.md`). Warstwa orkiestracji (`run_asset.py`, notebook wykonawczy, `iterators/`, ledgery `oos_read_ledger.jsonl`, `pretrain_universal.py`) istnieje wyłącznie na gałęzi badawczej.

## 1. Werdykt

**Werdykt: TAK — rdzeń systemu ML nie wykazuje look-ahead bias ani data leakage.**

Samodzielne sprawdzenie przeprowadzono zgodnie z procedurami matematycznymi, walidacyjnymi i inżynierskimi stosowanymi jako standard branżowy przy budowie systemów machine learning dla szeregów czasowych i strategii ilościowych.

Sprawdzenie miało charakter adwersarialny i zostało wykonane przez trzy niezależne agenty kontrolne. Zweryfikowano około 30 mechanizmów ochronnych, ograniczeń i warunków fail-closed.

### Wynik sprawdzenia

* liczba wykrytych problemów klasy `CRITICAL`: **0**,
* brak dostępu danych OOS do procesu tworzenia cech, etykiet, modeli i parametrów decyzyjnych,
* brak mieszania zbiorów Train, Validation i OOS,
* mechanizmy ochronne potwierdzone bezpośrednio w kodzie aktualnej wersji `HEAD`,
* cztery znane odstępstwa od idealnego eksperymentu one-shot zostały policzone, udokumentowane i jawnie oddzielone od wyników ML.

Wniosek dotyczy całego podstawowego łańcucha:

> **features → labels → CV → HPO → calibration → artifact → OOS execution**

---

## 2. Zakres i metoda sprawdzenia

Sprawdzenie nie opierało się na deklaracjach zawartych w komentarzach, dokumentacji ani opisach architektury.

Każda gwarancja została zweryfikowana bezpośrednio w wykonywalnym kodzie systemu, z odwołaniem do konkretnych plików i linii.

Kontrola obejmowała w szczególności:

* konstrukcję etykiet,
* granice czasowe Train, Validation i OOS,
* purge i embargo,
* mechanizmy cross-validation,
* generowanie cech,
* normalizację danych,
* HPO i feature search,
* warm-start modelu LSTM,
* kalibrację progów decyzyjnych,
* selekcję strategii,
* moment zamrożenia artefaktów,
* odczyty danych OOS,
* liczniki i ledgery odczytów,
* tryb fallback,
* historyczne eksperymenty i ograniczenia metodologiczne.

---

## 3. Twarde gwarancje zweryfikowane w kodzie

| Mechanizm kontrolny                                     | Dowód techniczny                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Etykiety nie sięgają do okresu OOS**                  | Obowiązuje warunek `t0 + H + embargo ≤ oos_start`, zabezpieczony assertami. XGB: `pipeline.py:320–324`. LSTM: `pipeline.py:178–182`. Wyjścia z symulacji barierowej są dodatkowo ograniczane do ostatniego bara zbioru Train w `simulate_trade`.                                                                                                                  |
| **Foldy CV nie przecinają granicy OOS**                 | Każdy scorer działający po stronie Train posiada warunki fail-closed. Granica assertu odpowiada rzeczywistemu oknu przetwarzania silnika. XGB: asserty fail-closed w HPO (`pipeline.py:848`) oraz w kalibracji punktu pracy (`pipeline.py:933`); replay silnika do `hi` wykonuje `op_grid_scores` (`pipeline.py:802–803`). LSTM wykorzystuje zakres do `hi + H`, a assert kontroluje dokładnie tę granicę: `model.py:201, 298`; `feature_search.py:288`. |
| **Features są przyczynowe**                             | Wszystkie okna czasowe są trailing. Stosowane są wyłącznie operacje `rolling`, `EWM` oraz `shift ≥ 1`. Walidator DSL propozycji (mechanizm podsystemu LSTM: `lstm/features.py:276–341`) posiada whitelistę operatorów historycznych i odrzuca `shift < 1`; XGB nie posiada DSL — search wybiera wyłącznie z zamkniętego rejestru zaimplementowanych cech. Kontekst `1d` i `1w` jest dołączany przez `merge_asof(direction="backward")` według czasu zamknięcia już zakończonego bara: `pipeline.py:536–548`. |
| **Normalizacja nie korzysta z danych przyszłych**       | W pipeline LSTM statystyki każdego foldu są obliczane wyłącznie na danych wcześniejszych niż `val_lo − embargo`: `pipeline.py:224–239`. Statystyki całego Train są używane dopiero przy budowie finalnego artefaktu przeznaczonego do OOS.                                                                                                                        |
| **Warm-start backbone’u korzysta wyłącznie z Train**    | Pooled panel jest budowany wyłącznie z wierszy należących do masek Train. Dane przechodzą purge, a etykiety są ograniczone do prawidłowego horyzontu: `pretrain_universal.py:74–99`. Dane OOS nigdy nie są używane do utworzenia checkpointu.                                                                                                                     |
| **Brak sprzężenia OOS z decyzjami modelu**              | Artefakt modelu, próg `θ` i zestaw cech są zamrażane przed odczytem `D9/L9`. Kalibracja i floor są wyznaczane wyłącznie na Train-OOF. `op_select` jest funkcją czystą i nie posiada dostępu do plików wynikowych. Żaden czytnik `oos_metrics` nie wpływa na selekcję modelu ani parametrów. Jedyną gałęzią zależną od wyniku OOS jest jawnie wydzielony fallback. |
| **Odczyty OOS są kontrolowane i policzone**             | Oba entrypointy `run_asset` posiadają choke pointy blokujące odczyt bez otwartej epoki eksperymentalnej (XGB `run_asset.py:43–46`, LSTM `run_asset.py:44–47`; backstop w `iterators/oos_ledger.py:136–154`). Każdy odczyt OOS jest zapisywany w committed ledgerze (`{xgb,lstm}/data/oos_read_ledger.jsonl`) wraz z licznikiem `read_count`.                      |
| **Selekcja parametrów odbywa się przed OOS**            | HPO, feature selection, kalibracja progów, wybór artefaktu i konfiguracja wykonawcza korzystają wyłącznie z danych Train lub Train-OOF. OOS nie uczestniczy w optymalizacji ani rankingu kandydatów.                                                                                                                                                              |
| **Wyniki ML są oddzielone od benchmarków wykonawczych** | Wynik właściwego modelu i tryb fallback posiadają odrębne pola `result_mode`. Fallback ma `trades=0` oraz `PF=None`, dzięki czemu nie może zostać przedstawiony jako wynik strategii ML.                                                                                                                                                                          |

---

## 4. Brak look-ahead bias i data leakage w rdzeniu pipeline’u

Na podstawie przeprowadzonych kontroli można stwierdzić, że dane przyszłe nie wpływają na:

1. generowanie cech,
2. konstrukcję etykiet,
3. podział foldów,
4. ocenę kandydatów w CV,
5. search cech,
6. HPO,
7. normalizację,
8. kalibrację progów,
9. wybór modelu,
10. budowę finalnego artefaktu,
11. decyzje wykonywane w okresie OOS.

Istotne jest również to, że zabezpieczenia nie ograniczają się do logicznych założeń architektury. System posiada aktywne asserty i mechanizmy fail-closed, które zatrzymują proces w przypadku naruszenia granic czasowych.

Oznacza to, że prawidłowość pipeline’u jest egzekwowana wykonywalnie, a nie jedynie opisana deklaratywnie.

---

## 5. Cztery jawne ograniczenia metodologiczne

Poniższe elementy nie stanowią leakage w podstawowym pipeline ML. Nie odpowiadają jednak idealnemu eksperymentowi typu strict one-shot, dlatego muszą być ujawniane przy prezentacji wyników.

> Szerszy raport spójności (`Raport_Spojnosci_Badan.md` §4) deklaruje **osiem** ograniczeń badania; poniższe **cztery** to proceduralny podzbiór tego samego zbioru — numeracje obu list są niezależne.

### 5.1. Fallback HODL jest wybierany po obserwacji liczby transakcji

Przełączenie na tryb:

> „kup pierwszego dnia OOS i trzymaj do końca okresu”

następuje dopiero po stwierdzeniu, że model nie wykonał żadnej transakcji w całym oknie OOS.

Taki fallback nie jest możliwy do wykonania ex ante, ponieważ decyzja o jego aktywacji wykorzystuje informację, że w całym okresie wystąpiło `0 trades`.

Z tego powodu zapieczętowana epoka przechowuje fallback jako osobny tryb:

* oddzielny `result_mode`,
* `trades = 0`,
* `PF = None`,
* brak kwalifikowania fallbacku jako wyniku ML.

Fallback jest wyłącznie benchmarkiem trybu wykonawczego. Nie może być łączony z wynikiem strategii ML ani przedstawiany jako jej rezultat.

Interfejs prezentacyjny zachowuje to rozdzielenie.

---

### 5.2. One-shot obowiązuje na poziomie epoki, a nie całego projektu

Okno OOS obejmujące okres `2024–2026` (XGB: `2024-01-02` – `2026-05-29`; LSTM: `2024-01-01` – `2026-04-30`) było odczytywane wielokrotnie w kolejnych epokach. **Stan po epoce `2026-07-golden-v5` (2026-07-19), wprost z ledgerów:** w tej epoce wykonano 588 odczytów dla uniwersum XGB (498 tickerów) i 495 dla LSTM (495 tickerów); skumulowany `read_count` na ticker wynosi **4–9 (średnia 5,18) dla XGB** i **4–6 (średnia 4,00) dla LSTM**.

Konsekwencja, którą trzeba nazwać wprost: zapieczętowany wynik **nie jest** pojedynczym, dziewiczym odczytem okna OOS — jest odczytem warunkowanym wiedzą z wcześniejszych odczytów tego samego okresu. Dyscyplina polega tu nie na tym, że odczyt był jeden, lecz na tym, że **każdy odczyt jest policzony i zapisany**: nieodnotowany ponowny odczyt byłby nie do odróżnienia od przebierania w wynikach OOS. Dokładna liczba odczytów każdego tickera jest zapisana w ledgerach jako `read_count` i pokazana na stronie Integrity.

Pomiędzy poszczególnymi epokami metodologia była poprawiana. Zmiany obejmowały przede wszystkim:

* usuwanie wykrytych źródeł biasu,
* wzmacnianie purge i embargo,
* uszczelnianie bramek,
* dodawanie assertów,
* eliminowanie możliwości poluzowania warunków walidacyjnych.

Nie oznacza to leakage wewnątrz aktualnie zamrożonej epoki. Oznacza jednak, że zapieczętowany wynik jest warunkowany wiedzą uzyskaną podczas wcześniejszych odczytów tego samego okresu OOS.

Projekt nie udaje zatem laboratoryjnie czystego, pierwszego i jedynego odczytu. Dyscyplina eksperymentalna polega tutaj na:

* liczeniu każdego odczytu,
* rozdzielaniu epok,
* zamrażaniu konfiguracji przed kolejnym odczytem,
* zapisywaniu historii metodologii,
* ujawnianiu zależności od wcześniejszych iteracji.

---

### 5.3. Bias inicjalizacji LSTM i inflacja gainów w searchu XGB

Backbone LSTM w procesie warm-startu widział wiersze należące do zbioru Train pochodzące z różnych granic foldów.

Nie powoduje to dostępu do OOS, ale może wpływać na absolutny poziom wyników osiąganych w poszczególnych foldach.

Praktyczna interpretacja jest następująca:

* rankingi kandydatów mogą być używane,
* względne porównania pozostają informacyjne,
* wartości absolutne metryk powinny być interpretowane ostrożniej.

W przypadku XGB search prowadzony na większym supersetcie konfiguracji może powodować inflację raportowanych gainów w fazie poszukiwania.

Oba ograniczenia zostały opisane w `docs/FEATURE_SEARCH_METHODOLOGY.md` (gałąź badawcza `to_give_up_and_show`; bias warm-startu: linie 79–88, inflacja gainów: 89–99). Na gałęzi `Stable_Presentable_Version` odpowiednikiem jest `docs/METHODOLOGY.md` (§5 oraz tabela ograniczeń).

Konfiguracje obciążone tymi mechanizmami nie są bezpośrednio wdrażane do produkcyjnego artefaktu bez ponownej walidacji.

---

### 5.4. Survivorship bias uniwersum i historyczny wyjątek pilotażowy

Uniwersum historyczne jest budowane na podstawie obecnego składu indeksu S&P 500.

Powoduje to klasyczny survivorship bias, ponieważ historyczna analiza nie zawiera wszystkich spółek, które:

* wcześniej należały do indeksu,
* zostały z niego usunięte,
* zostały przejęte,
* zbankrutowały,
* utraciły wymagane kryteria kapitalizacji lub płynności.

Dodatkowo w historycznym pilocie jeden kandydat feature został oceniony po obserwacji OOS. Dotyczyło to feature `906` (`xtf_zscore_1h_vs_1d`).

Przypadek został udokumentowany w raporcie pilotażowym `docs/FEATURE_PILOT_REPORT.md` (wyłącznie gałąź badawcza — brak odpowiednika na tej gałęzi).

Aktualny kontrakt steera zabrania takiego postępowania. Obecnie dowody dopuszczające feature do dalszego procesu mogą pochodzić wyłącznie z Train-CV.

---

## 6. Ustalenia klasy MINOR

Sprawdzenie wykazało również kilka problemów klasy `MINOR`. Nie wpływają one na zamrożone wyniki sealed epoki, ale stanowią listę technicznego hardeningu dla przyszłych wersji.

| Problem                                                                                    | Znaczenie                                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Scratch exemption sprawdza obecność zmiennej środowiskowej zamiast poprawności ścieżki** | Mechanizm powinien walidować konkretną ścieżkę i jej zakres, a nie jedynie fakt ustawienia zmiennej.                                                                                                         |
| **Ledger jest aktualizowany po zapisie wiersza wynikowego**                                | Awaria procesu pomiędzy zapisem wyniku a zapisem ledgera może pozostawić niepełny ślad audytowy. W przyszłości zapis powinien być transakcyjny lub wykonywany w odwrotnej kolejności z mechanizmem recovery. |
| **Walidator DSL czyta bary z okresu OOS podczas testu skończoności**                       | Dane są używane wyłącznie do kontroli strukturalnej, np. wykrywania `NaN` lub `inf`, a nie do oceny wyników czy selekcji. Mimo braku wpływu decyzyjnego lepiej ograniczyć walidację wyłącznie do Train.      |
| **Early stopping może obserwować validation wewnątrz foldu w zimnej ścieżce HPO**          | Jest to lokalny peeking w ramach procedury strojenia danego foldu. Nie narusza granicy OOS, ale może lekko optymistycznie wpływać na ocenę konfiguracji w searchu.                                           |

Aktualna epoka posiada status `FROZEN`, dlatego poprawki nie są wprowadzane retroaktywnie.

Zmiana kodu po zobaczeniu wyników naruszałaby zasadę niezmienności zamrożonego eksperymentu. Elementy te powinny zostać poprawione dopiero w kolejnej, jawnie oznaczonej epoce metodologicznej.

---

## 7. Zasady prezentowania wyników

W prezentacji, konsoli i raportach należy bezwzględnie rozdzielać następujące kategorie:

| Kategoria                       | Sposób prezentacji                                                               |
| ------------------------------- | -------------------------------------------------------------------------------- |
| **Wynik ML**                    | Wynik zamrożonego artefaktu, działającego na OOS bez dostrajania po odczycie.    |
| **Fallback HODL**               | Osobny benchmark wykonawczy; nigdy jako wynik ML.                                |
| **Train-CV / Train-OOF**        | Wyniki służące do selekcji, kalibracji i oceny stabilności.                      |
| **Sealed OOS**                  | Końcowa obserwacja generalizacji dla danej epoki.                                |
| **Historyczne epoki**           | Wyniki oznaczone numerem epoki i liczbą wcześniejszych odczytów OOS.             |
| **Ograniczenia metodologiczne** | Widoczne na stronie `Integrity`, bez ukrywania lub agregowania z główną metryką. |

Na stronie `Integrity` powinny być jawnie prezentowane co najmniej:

* numer epoki,
* status `FROZEN`,
* `read_count`,
* zastosowany purge,
* długość embargo,
* horyzont etykiety `H`,
* granica `oos_start`,
* tryb wyniku,
* informacja o fallbacku,
* sposób inicjalizacji modelu,
* status survivorship bias,
* lista znanych ograniczeń,
* lista ustaleń `MINOR`,
* wersja kodu lub commit `HEAD`.

---

## 8. Konkluzja

Brak look-ahead bias i data leakage w łańcuchu:

> **features → labels → CV → HPO → calibration → artifact**

został potwierdzony przez bezpośrednią analizę kodu, aktywne asserty, warunki fail-closed oraz niezależne sprawdzenie przeprowadzone zgodnie z matematycznymi i inżynierskimi standardami branżowymi.

Dane OOS nie wpływają na tworzenie cech, etykiet, dobór modeli, strojenie parametrów, kalibrację ani budowę finalnego artefaktu.

Jednocześnie projekt jawnie ujawnia miejsca, w których procedura odbiega od laboratoryjnego ideału:

* fallback aktywowany po stwierdzeniu braku transakcji,
* wielokrotne epoki odczytu tego samego okresu OOS,
* bias inicjalizacji warm-startu LSTM,
* inflację gainów w searchu XGB,
* survivorship bias uniwersum,
* historyczny wyjątek pilotażowy.

Ograniczenia te są policzone, zapisane w ledgerach, opisane w dokumentacji i widoczne w konsoli.

To rozdziela badanie metodologicznie uczciwe od prezentacji upiększonej: system nie ukrywa odstępstw od ideału, lecz określa ich zakres, wpływ i miejsce w procesie decyzyjnym.
