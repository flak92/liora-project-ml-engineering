# KNOW-HOW & BEST-PRACTICE — poszukiwanie cech OHLCV dla XGB ∥ LSTM (v1)

> Przenośny destylat metodologii z dwóch projektów feature-search na OHLCV (Golden Calibration XGB +
> universal LSTM). **Cel: dało się z tego poprowadzić NOWY projekt na Cryptoassets** — bez powtarzania
> tych samych błędów. To nie jest kopia kodu, tylko **zasady + dokładne pokrętła, które zadziałały**, z
> odsyłaczami `plik:linia` do miejsc w tym repo. Liczby są odczytem realnego kontraktu, nie ilustracją.
>
> Rozdział **8** jest dedykowany adaptacjom pod krypto. Rozdziały 1–7 to rdzeń uniwersalny; 9–10 to
> checklista i przepis startowy. Język: polski, terminy techniczne po angielsku.

---

## 1. Teza — co w ogóle robimy

**Nie szukamy „najlepszych wskaźników".** Szukamy **najmniejszego zbioru niezależnych mechanizmów
informacyjnych OHLCV, których reprezentanci przechodzą uczciwą walidację i przenoszą się między
aktywami.** Trzy konsekwencje, które trzeba przyjąć zanim napiszesz pierwszą linijkę:

1. **Lejek jest własnością danych, nie celem.** Przebieg `26 → 11 → 9 → 2 → 1` (provisional → A1 →
   A1×A2×B → retained → unique feature) to migawka jednego runu. Świeży panel może dać `30 → 7 → 2 → 0`
   i być równie poprawny. Nie „dąż do wyższej liczby na końcu".
2. **Uczciwy null-result jest wynikiem naukowym.** „Edge nieistotny po korekcie multiple-testing" to
   test poprawności rurociągu, nie porażka. Pusty zbiór potwierdzonych cech = poprawny wynik.
3. **Cecha nie jest „dobra" bo wysoko zrankowała.** Jest potwierdzona tylko wtedy, gdy przeżyła wybór
   i osąd danymi, które w wyborze nie brały udziału, i pobiła maksimum, jakie samo przeszukiwanie
   produkuje.

**Dwa estymatory o różnej indukcji na tym samym materiale.** XGB (drzewo, wejście tabelaryczne) i LSTM
(sekwencja, wejście oknowe) uczą się z tej samej etykiety i zdarzeń. Jeśli sygnał widzi tylko jeden —
jest własnością modelu, nie danych. Jeśli widzą oba — sprawdzasz jeszcze, czy nie mylą się w tych samych
miejscach (korelacja błędów).

SSOT tej filozofii w tym repo: [`docs/SMART_METHODOLOGY.md`](docs/SMART_METHODOLOGY.md).

---

## 2. Fundament danych i integralność — ZANIM dotkniesz modelu

Najczęstsza porażka projektu ML-tradingowego nie jest w modelu, tylko tu. **Wada niewykryta na tym
poziomie staje się „cechą" i wygląda jak sygnał.**

- **Kontrakt czasu.** Każdy bar niesie `bar_open_ts`, `bar_close_ts`, `available_at_ts = bar_close_ts`.
  Decyzja: `decision_ts = bar_close_ts(t0)`, wejście opóźnione `entry_idx = t0 + ENTRY_LAG`, gwarancja
  STRICT `decision_ts < entry_fill_ts`. Cechy czytają wyłącznie **domknięte** bary (`shift ≥ 1`).
- **Granica Train/OOS czytana ZERO razy** do certyfikacji. W tym repo: Train ≤ `2023-12-29`, OOS od
  `2024-01-02`, `oos_reads = 0`, `embargo_bars = 35`, `label_horizon_bars = 24`
  ([`config/contract/label_contract.json`], odczyt: `feature_discovery_contract.json:data_boundary`).
- **Purge + embargo** przy walk-forward: zdarzenie trafia do treningu tylko gdy `t1 < test_start`
  (etykieta żyje do `t1`, więc purge obejmuje cały jej horyzont), plus embargo. `OOF ⊂ fold`.
- **Zakazy twarde:** forward-fill, backfill, bary syntetyczne. Luki wobec kalendarza **raportowane, nigdy
  łatane** (załatana luka = cecha, której nie było).
- **Etykieta = triple-barrier od opóźnionego entry.** Asymetryczne bariery ATR: TP/SL na HIGH/LOW barów
  pozycji, pionowa H barów; `Y ∈ {−1,0,+1}`. Bariery liczone od `open(t0+ENTRY_LAG)`, nie od `close(t0)`
  (liczenie od ceny niedostępnej w chwili decyzji zawyża wynik). W tym repo `TB_ATR_TP=2, TB_ATR_SL=1`,
  reward:risk `b = TP/SL` ([`config/xgb.json`]).
- **Determinizm.** `SEED=42`, piny wątków jako **piny wyniku** (nie wydajności), byte-identyczna
  reprodukcja. Raport **generuje się** z zapieczętowanych artefaktów, nie jest przepisywany.

Źródła: [`config/contract/`](config/contract/), [`docs/SMART_METHODOLOGY.md`](docs/SMART_METHODOLOGY.md) §0–3.

---

## 3. Optuna — zasada działania (tuner), i żelazne reguły

**Jak to realnie zestawione (identyczny wzór dla XGB i LSTM):**

```python
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),          # deterministyczny przy seedzie
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=WARMUP))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
```

- XGB: [`xgb/src/pipeline.py:964`](xgb/src/pipeline.py); sealed model `N_TRIALS = 80`, `cv_folds = 4`
  (inner purged WF), warmup z `search_space.objective.pruner_warmup`.
- LSTM: [`lstm/model.py:237`](lstm/model.py); `n_trials = 20`, `pruner_warmup_steps = 2`
  ([`config/lstm.json`] / `lstm HPO`).

**TPESampler** = Tree-structured Parzen Estimator: buduje dwa rozkłady gęstości nad wcześniejszymi
trialami (dobre vs złe wyniki) i próbkuje tam, gdzie stosunek gęstości „dobrych" jest wysoki. Lepszy niż
random/grid przy budżecie kilkudziesięciu triali. **MedianPruner** ubija trial, którego pośredni wynik
jest poniżej mediany dotychczasowych na tym kroku — po `n_warmup_steps` rozgrzewki (żeby nie ubijać
zbyt wcześnie). Seed w samplerze + piny wątków = **powtarzalny wynik HPO**.

**Żelazne reguły (łamiesz jedną — wynik jest nieważny, cicho):**

| # | reguła | dlaczego |
|---|---|---|
| a | HPO **wyłącznie na Train inner-CV** (purged WF); nigdy na foldach ocenianych ani na holdoucie | strojenie na ocenianym oknie to ukryte dopasowanie do wyniku |
| b | cel tunera **klasyfikacyjny** (macro-F1 / AUC-PR), **NIGDY PnL/Sharpe** | tuner na PnL optymalizuje backtest, nie zdolność predykcyjną (backtest-fitting) |
| c | podczas **feature-search** HPO tunuje **tylko core**, cechę oceniasz przy zamrożonym modelu | inaczej mierzysz strojenie, nie cechę (patrz LSTM „lekki ustalony model", §5) |
| d | **każdy trial liczy się do budżetu multiplicity** (`n_trials` w PBO/DSR), także odrzucony | im więcej hipotez sprawdziłeś, tym łatwiej o przewagę, której nie ma |
| e | próg progu decyzji `p*` wyznaczaj na **anchor OOF**, nie na ocenianym oknie | jw. |
| f | **pin seeda + wątków**; zapisz `contract_hash`, wersje pakietów, `env_manifest` | bez tego „ten sam eksperyment" nic nie znaczy |

**Przestrzeń przeszukiwania nie jest absolutna — kalibruje się do geometrii danych** (patrz §4, `g_rel`).
To jest różnica między „narzuconą na wszystkie aktywa" a „wyprowadzoną z każdego aktywa".

---

## 4. XGB — uczeń drzewiasty

- **Wejście:** wiersz cech point-in-time `[1 × F]`. Rdzeń 1h (ids 1–17) **zamrożony, nigdy nie
  przeszukiwany** — to baseline; przeszukiwane są wyłącznie cechy opt-in (w tym repo 45 kandydatów / 12
  rodzin).
- **Viability floor = bramka DOPUSZCZALNOŚCI, nie selektor.** Model musi w ogóle umieć się uczyć zanim
  ocenisz cechę: `split_nodes ≥ 20`, `pred_std ≥ 0.005`
  ([`scripts/feature_utility.py:72`](scripts/feature_utility.py)). Floor **nie rankuje** i nie wybiera
  zwycięzcy — tylko wpuszcza albo odrzuca.
- **Przestrzeń Optuny skalibrowana do danych (kluczowy trik).** `gamma` i `min_child_weight` są progami
  w **jednostkach hessianu** (dla `binary:logistic` wiersz wnosi `w·p(1−p)`). Zamiast wartości
  bezwzględnych używamy **hessian-relative** `g_rel`: ta sama wartość znaczy to samo na każdym aktywie,
  bo `label_uniqueness_weight` kurczy efektywną próbę ~18× i absolutny `gamma` byłby o rząd wielkości za
  agresywny. Zmierzone: `g_rel=0` → 979–1084 split nodes, `0.005` → 66–260, `0.02` → 3–19 — spójnie na
  czterech aktywach i dwóch rozmiarach folda. Źródło z pomiarami:
  [`config/xgb_search_space_v2.json`](config/xgb_search_space_v2.json) `_meta`. **To jest sedno „smart
  calibration": przestrzeń wyprowadzasz z danych, nie narzucasz.**
- **Etykieta + meta-labeling.** Asymetryczny ATR triple-barrier (`TB_ATR_TP=2, TB_ATR_SL=1`); przed
  wtórnym XGB stoi `ENTRY_GATE` — stały, Train-only, przyczynowy filtr pierwotny (meta-labeling), który
  wpuszcza tylko istotne setupy, żeby wtórny model nie tonął w szumie ([`config/xgb.json`]).
- **Cechy wchodzą z rejestru rodzin** (`config/feature_families_xgb.json` + `config/feature_registries/`),
  nie z ad-hoc listy w notatniku.

---

## 5. LSTM — uczeń sekwencyjny

- **Wejście:** okno `[T_SEQ × F]` z tych samych kolumn co XGB. Architektura (hidden, num_layers, seq_len,
  dropout, lr) + **frozen Train-fit scaler** (mean/std per cecha) w [`lstm/data/universal_backbone.json`].
- **Universal backbone + per-asset overrides.** Jeden LSTM trenowany na wszystkich aktywach na wspólnym
  zbiorze cech; potem każdy asset dobiera swoje rozszerzenia (`selected_optional_ids`,
  [`lstm/data/per_asset_feature_overrides.json`]).
- **Trik oceny cechy: LEKKI, USTALONY LSTM.** Cechę oceniasz core-only Train CV AUC-PR **lekkim, o
  ustalonych hiperparametrach modelem** — żeby wynik odzwierciedlał **cechę, nie HPO**
  ([`lstm/feature_search.py:5`](lstm/feature_search.py)). To odpowiednik reguły 3c dla sekwencji.
- **Forward selection pod overfit gate, score PENALIZED.** Cecha wchodzi tylko gdy poprawa jest REALNA:
  `score = mean CV AUC-PR − complexity_penalty · n_added`, plus `min_feature_gain` i `min_fold_win_frac`
  ([`lstm/feature_search.py:323`](lstm/feature_search.py)). Purged walk-forward folds.
- **Ranking ablacyjny (AP-drop):** ważność cechy = ile spada validation AUC-PR, gdy ją usuniesz
  (mierzone na Train, nigdy na OOS; [`lstm/data/universal_feature_ranking.json`]).
- **Operating point** θ (próg wejścia), λ (Kelly fraction), kierunek — fitowany na Train **out-of-fold**
  log-growth ([`lstm/data/calibration_records_v5.json`]). Uwaga: λ bywa `None` w epoce all-in — operating
  point jest **uczciwy wobec swojej epoki**, nie miesza wartości między epokami.

---

## 6. Optimized feature search — drabina i nulle

Poszukiwanie cech jest **rurociągiem z bramą**, nie czynnością przy okazji treningu. Drabina (Rung):

```
rodziny (taksonomia mechanizmu)
  → Rung 3  marginal utility     — czy cecha poprawia model, który potrafi się uczyć
  → Rung 4  cross-fit            — czy wybór przeżywa dane, które go nie wybrały (rotujące foldy)
  → Rung 5  procedure-level null — edge > maksimum, które samo przeszukiwanie produkuje  (A1 × A2 × B)
  → Rung 6  survivor own-null    — ile survivor wart pod WŁASNYM tuned modelem  (może tylko DEGRADOWAĆ)
  → Rung 8  family transfer      — które rodziny przenoszą się między aktywami  (statusy, minimal set)
  → JEDEN odczyt OOS             — certyfikacja
```

**Nulle (Rung 5) — serce uczciwości.** Statystyka akceptacji na permutowanych blokach; wartości
z kontraktu ([`config/feature_discovery_contract.json:max_null`], `config/contract/multiplicity_contract.json`):
- `M = 50` permutacji, `α = 0.10`, pass gdy `b ≤ 4` (`b = #{null_stat ≥ real_stat}`),
  `p_mc = (1+b)/51 ≤ 5/51`, futility stop przy `b = 5`.
- **A1 marginal** — łapie *search-inflation* (edge to tylko maksimum szerokiego przeszukiwania).
- **A2 regime** — grupowana permutacja (segmenty); łapie zależność od reżimu.
- **B conditional** — residual conditional (Ridge); łapie zależność od sprzężenia z core.
- `confirmed = A1 ∩ A2 ∩ B`, **fail-closed**: brakujący null nie może potwierdzić stabilności (nie
  „benefit of the doubt"). Kanoniczna definicja survivora w jednym miejscu:
  [`scripts/rung5_verdict.py`](scripts/rung5_verdict.py) — importują ją i raport, i maszyna stanów.

**Acceptance (Rung 4):** `min_rotations = 2`, `min_median_confirmation_delta = 0.004`,
`complexity_penalty = 0.004`, statystyka `T = max(median_delta − 0.004)` po eligible arms, majority-positive.

**Reprezentant one-SE + stability selection:** z klastra korelacyjnego bierzesz najprostszą cechę w
granicy jednego błędu standardowego od najlepszej; utrzymujesz tylko cechy wybierane powtarzalnie
(bootstrapy, próg częstości). To kontroluje redundancję i multiple-testing **na etapie doboru**, zanim
policzysz PBO/DSR.

**Rung 8 (family transfer)** — wzorzec czystego agregatora: [`scripts/family_transfer.py`](scripts/family_transfer.py).
Czyta zamrożone artefakty (crossfit, nulle, rung6, compiled), normalizuje `unit` (int/str per arm),
deduplikuje flat+hierarchical → jeden unique representative, nadaje każdej rodzinie **dokładnie jeden
status** (`PANEL_STABLE / ASSET_CONDITIONAL / CROSSFIT_UNSTABLE / SEARCH_INFLATED / REGIME_DEPENDENT /
CORE_CONDITIONAL_FAILURE / TUNING_DEPENDENT / NO_POSITIVE_UTILITY / INSUFFICIENT_EVIDENCE`), liczy
`minimal_panel_family_set` (set-cover nad potwierdzonymi `asset × fold`, nie ranking), i taksonomię
(KEEP/REVIEW_SPLIT/REVIEW_MERGE). **Zero treningu, zero OOS, zero duckdb** — dowiedzione self-testem.

---

## 7. Multiplicity, determinizm, uczciwość

- **Każdy wariant jest próbą.** `n_trials` zasilające PBO/DSR = `COUNT(trials_ledger)` kampanii, **nie 1**.
  Pętla bez rejestru prób unieważnia PBO/DSR po cichu.
- **Tożsamość recipe = hash.** Edycja mapy rodzin / przestrzeni cech = **nowa epoka wyszukiwania** (nowy
  `contract_hash`), nigdy cicha zmiana. `sample_sha256` wchodzi w `contract_hash` — zmiana zbioru aktywów
  unieważnia zamrożone artefakty.
- **Raport = odczyt artefaktów** (A0.2): każda liczba w raporcie jest odczytem zapieczętowanego artefaktu,
  nie przepisana ręcznie. Regeneracja ⇒ diff = 0.
- **Nazewnictwo uczciwe:** `pbo_like_diagnostic` / `dsr_diagnostic` dopóki założenia kanonicznych testów
  niespełnione; opis rodowodu wyniku zamiast „dziewiczy OOS".
- **Wzór czystej funkcji raportującej:** [`scripts/family_transfer.py`](scripts/family_transfer.py) +
  [`scripts/family_transfer_selftest.py`](scripts/family_transfer_selftest.py) (registry · determinizm ·
  fail-closed · dedup · **leakage: zero importów trenujących / bar store / OOS** · snapshot parity).
  `OOS READ COUNT = 0` jest testowalne statycznie.

---

## 8. ADAPTACJE POD KRYPTO (nowy projekt)

Metodologia przenosi się w całości; zmienia się **profil danych** i **mapa rodzin**. Traktuj poniższe jak
listę „co przeliczyć, zanim skopiujesz kontrakt".

### 8.1 Sesja i czas
- **24/7, brak RTH i kalendarza świąt, brak early-close.** Znika rodzina `session_position` (nie ma
  „pozycji bara w sesji"). Bary są gęste i ciągłe → to samo `T_SEQ` znaczy inny horyzont kalendarzowy niż
  na 1h akcjach; dobierz `seq_len` do horyzontu, nie do liczby barów.
- **Czas UTC natywnie, brak DST** — cała klasa wady D6 (przejścia DST, 23/25-barowe doby) znika.
- **Nowe „szwy" czasowe zamiast weekendów:** godziny fundingu (perp co 8 h), okna maintenance giełdy,
  rollover kontraktów kwartalnych. To są kandydaci na flagi/cechy sesyjne krypto.

### 8.2 Zdarzenia „korporacyjne" krypto
- Brak splitów/dywidend, **ale** są: **forki łańcucha, migracje/redenominacje tokenów, wrapped↔unwrapped,
  delisting/relisting, zmiana pary bazowej (USDT→USDC), zmiana kontraktu perp.** To jest odpowiednik
  guardrailu „skok ceny bez wpisu w rejestrze zdarzeń = HARD STOP". Zbuduj rejestr `CHAIN_ACTIONS`
  analogiczny do `CORP_ACTIONS` i bramę fail-closed na skok bez wpisu.

### 8.3 Mikrostruktura i płynność
- **Perp vs spot** to dwa różne instrumenty na ten sam bazowy — nie mieszaj bez jawnego basis.
- **Nowe rodziny cech specyficzne dla krypto:** `funding_rate` (poziom + zmiana + kumulacja), `basis`
  (perp − spot), `open_interest` (poziom + Δ), `long_short_ratio`, `taker_buy_sell_imbalance`,
  order-book imbalance (gdy masz L2). Te rodziny nie istnieją na akcjach i często niosą najwięcej sygnału.
- **`liquidity_tier` rozjeżdża się skrajnie** (BTC/ETH vs mid-cap alty vs mikro). Cechy mikrostrukturalne
  (bid-ask bounce, imbalance) dopuszczaj **tylko przy wysokiej płynności** — na niepłynnym alcie mierzysz
  szum kwotowania, nie rynek (analog wady D9).

### 8.4 Zmienność i QC
- **Reżimy zmienności ekstremalne; „violent bars" bywają PRAWDZIWE** (flash-move, likwidacje kaskadowe) —
  nie zawsze błąd danych. Przelicz pas guardrailu `JUMP_SOFT..JUMP_HARD` na **rozkład krypto** (kwantyle
  tego store'u), inaczej wytniesz realny rynek albo wpuścisz śmieci. Pas nietykalny (prawdziwy ruch) jest
  szerszy niż na akcjach.
- **Zero-volume / duplikaty** częstsze na alt-parach i podczas maintenance — QC (analog D4/D5) tym
  ważniejsze.

### 8.5 Dobór panelu i historia
- **Krótka historia altów** → więcej rodzin `INSUFFICIENT_EVIDENCE`; nie ratuj tego luźniejszym progiem.
- **Panel rozpinający osie**, nie wiele podobnych: BTC, ETH, duży L1 (SOL), L2 (ARB/OP), DeFi blue-chip
  (UNI/AAVE), mem (DOGE), para stable jako kontrola. `A6`/`A10` z rejestru = celowo najtrudniejsze
  (krótka historia + niska płynność) — test, czy metoda **odmawia** wniosku przy zbyt małej próbie.
- **Survivorship:** uniwersum krypto zmienia się szybko (delisting/nowe listingi). Zamroź uniwersum na
  starcie kampanii i hashuj — zmiana = nowa `campaign_id`.

### 8.6 Koszty
- **Przelicz `p_min = (SL + koszty_rt) / (TP + SL)` na krypto:** taker/maker fee + **funding** (perp ma
  ciągły koszt utrzymania pozycji, którego equity spot nie ma) + slippage (funkcja `liquidity_tier`).
  Koszty ustalają PRÓG opłacalności, nie tylko obniżają wynik — zmiana kosztów przesuwa bramkę i
  unieważnia dopuszczenia stref/cech policzone przy poprzedniej wartości.

### 8.7 Determinizm
- **LSTM na GPU jest niedeterministyczny** (cuDNN atomics). Opcje: (a) pin `torch.use_deterministic_algorithms(True)`
  + `CUBLAS_WORKSPACE_CONFIG`, (b) trenuj na CPU (wolniej, byte-deterministyczne), (c) zaakceptuj i
  **mierz rozrzut** między seedami zamiast udawać byte-parity. XGB pozostaje byte-deterministyczny.

### 8.8 Mapa rodzin dla krypto (punkt startowy)
Zostaw: `momentum_return`, `volatility_level`, `volatility_regime`, `range_position`, `volume`,
`price_distance`, `trend_slope_ma`, `oscillator_rsi`, `macd`, `multi_tf_alignment`, `cross_tf_ratio`.
Wytnij: `session_position` (24/7). Dodaj: `funding`, `basis`, `open_interest`, `perp_flow`.
**Edycja mapy = nowa recipe identity (nowa epoka).**

---

## 9. Checklista „czego NIE wolno" (fail-closed)

- ❌ OOS do wyboru cech / progu / hiperparametrów.
- ❌ PnL / Sharpe jako cel tunera (backtest-fitting).
- ❌ SHAP jako selektor cech (importance ≠ przyczynowość; policzone na całości = leakage).
- ❌ correlation pruning **po** tym, jak kandydat zobaczył target.
- ❌ zmiana `α` / liczby permutacji / granicy OOS / viability floor **po** słabym wyniku.
- ❌ „najlepsza dostępna cecha" = potwierdzona.
- ❌ pusty wynik traktowany jako awaria.
- ❌ drugi, równoległy feature registry nieużywany przez procedure null.
- ❌ nazwanie przebiegu deweloperskiego (mały znany panel) certyfikacją.
- ❌ pętla bez rejestru prób (`trials_ledger`) → PBO/DSR to fikcja.
- ❌ raport z liczbami przepisanymi ręcznie zamiast odczytanymi z artefaktu.

---

## 10. Minimalny przepis startowy dla projektu crypto (pierwszy tydzień)

1. **Zamroź problem (Rung 0).** Uniwersum (panel rozpinający osie, hash), granica Train/OOS (`oos_reads=0`),
   etykieta (triple-barrier ATR, ENTRY_LAG), seedy, piny wątków, `contract_hash`.
2. **QC wad danych.** Rejestr wad (luki, zero-volume, duplikaty ts, `CHAIN_ACTIONS`, violent-bars-band
   przeliczony na krypto) + bramy fail-closed. **Wada niewykryta staje się cechą.**
3. **Zdefiniuj rodziny** (§8.8, z `funding`/`basis`/`OI`). Zamroź mapę hashem.
4. **Viability floor** (XGB: split_nodes/pred_std w skali hessianu; LSTM: lekki ustalony model do scoringu).
5. **Cross-fit confirmation** na rotujących foldach (`min_rotations=2`, penalized delta).
6. **Jeden null (A1)** na małym panelu (BTC/ETH/L1/L2/DeFi/mem), `M=50`, `α=0.10`, `b≤4`. Policz **uczciwy
   lejek**. Zatrzymaj się i przeczytaj wynik — może być `[]` i to OK.
7. **Dopiero potem** A2/B, Rung 6 (own-null), Rung 8 (family transfer), i na samym końcu — **jeden odczyt
   OOS**.

**Co skopiować z tego repo (wzorce, nie kod-1:1):**
- [`scripts/family_transfer.py`](scripts/family_transfer.py) — wzór czystego agregatora panelowego
  (read-only, deterministyczny, fail-closed, self-test, `OOS READ COUNT = 0`).
- [`scripts/rung5_verdict.py`](scripts/rung5_verdict.py) — jedna kanoniczna definicja survivora
  (A1∩A2∩B), importowana przez raport i maszynę stanów.
- [`config/contract/`](config/contract/) — split kontraktu na FROZEN (proof standard) vs ADMISSIBLE
  (przestrzeń hipotez) + strażnik pola-po-polu ([`engine/contract_patch.py`](engine/contract_patch.py)).
- [`config/xgb_search_space_v2.json`](config/xgb_search_space_v2.json) — wzór przestrzeni Optuny
  **kalibrowanej do geometrii danych** (`g_rel`).
- [`config/family_transfer_reporting.json`](config/family_transfer_reporting.json) — progi *raportowania*
  trzymane **poza** `contract_hash` (etykiety, nie proof standard).

---

## Provenance
Destylat z: repo `liora-project-ml-engineering` branch `methodology` (Golden Calibration XGB + Rung 8
Family Transfer) oraz siostrzanego narzędzia deliberacji XGB ∥ LSTM. Liczby (α=0.10, M=50, `b≤4`,
`p_mc=(1+b)/51`, viability 20/0.005, TB 2:1, embargo 35, horizon 24, funnel 26→11→9→2→1) są odczytem
kontraktu i snapshotu tego repo, nie ilustracją. **Wersja v1.** Zmiana metodologii = v2 (append-only).
