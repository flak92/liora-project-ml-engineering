# Podsumowanie projektu

Podczas budowania projektu i poszukiwania optymalnych features najważniejszym rezultatem nie okazało
się znalezienie jednej „najlepszej” konfiguracji. Istotniejszym wynikiem było stworzenie
metodologii, która nadaje kolejnym eksperymentom kierunek i pozwala automatycznie rozstrzygać, jaki
najmniejszy następny krok może jeszcze zmienić odpowiedź.

Dla każdego assetu powstaje indywidualny, matematyczny opis jego zachowania oparty na relacjach
OHLCV. Nie jest to dosłowne odtworzenie algorytmu całego rynku. Jest to przybliżenie warunkowej
logiki zmian ceny w przestrzeni stanów, które model potrafi rozpoznać i zweryfikować na danych
nieuczestniczących w wyborze.

W przypadku XGBoost oznacza to poszukiwanie regionów feature-space, w których decyzje modelu są
wystarczająco stabilne, aby działać jako filtr. Globalna trafność zbliżona do 50% sama w sobie nie
tworzy przewagi. Wartość może pojawić się dopiero wtedy, gdy model rozpoznaje konkretne stany, w
których jego opis danego assetu jest bardziej precyzyjny niż zwykle, oraz pozostaje nieaktywny poza
tymi stanami.

LSTM realizuje analogiczny cel z perspektywy sekwencji. Nie analizuje wyłącznie aktualnego punktu,
lecz trajektorię stanów prowadzących do niego. Przy dziennych danych OHLCV liczba niezależnych
sekwencji jest jednak ograniczona, dlatego model musi pozostać prosty, regularizowany i oceniany
konserwatywnie. Procedura wyprowadzona dla LSTM wymaga osobnej walidacji, ale zachowuje tę samą
drabinę: viability, transfer punktu operacyjnego, utility, confirmation, multiplicity control i
survivor-specific optimization.

Cały proces przypomina pracę specjalisty obserwującego rynek przez wiele lat: wykonującego kolejne
próby, zapisującego rezultaty i stopniowo budującego zbiór reguł działających w określonych
warunkach. Różnica polega na tym, że każda decyzja jest tutaj policzalna, reprodukowalna,
wersjonowana i sprawdzana według tego samego kontraktu.

Pobicie szerokiego rynku wyłącznie na podstawie historycznego OHLCV może wymagać wyjątkowo
sprzyjającego okresu, losowości albo dodatkowych informacji spoza samej ceny i wolumenu. Takimi
źródłami mogą być relacje cross-asset, przepływy kapitału, zmiany płynności, dane fundamentalne,
struktura zleceń lub sygnały dostępne dopiero w czasie rzeczywistym.

Zanim jednak system zacznie podejmować decyzje real-time, potrzebuje stabilnego fundamentu. Tym
fundamentem jest filtr, który nie próbuje przewidywać każdego ruchu, lecz rozpoznaje wyłącznie
sytuacje, które potrafi opisać wystarczająco spójnie i które przeżyły pełną procedurę niezależnego
potwierdzenia.

Najważniejszym rezultatem projektu nie jest więc obietnica pokonania rynku. Jest nim audytowalny
proces, który dla każdego assetu buduje własny matematyczny opis, wybiera stabilne features,
kalibruje zakresy parametrów do geometrii danych, zamraża reguły przed OOS i uczciwie pokazuje
zarówno miejsca, w których model działa, jak i te, w których powinien pozostać bezczynny.

To jest właściwy punkt wyjścia do systemu real-time: nie zbiór przypadkowych decyzji modelu, lecz
uporządkowany, automatycznie wykonywany i zweryfikowany filtr logiki rynku.

---

## Najważniejsza decyzja architektoniczna

> Branch `methodology` zawiera nie tylko opis Rungów, ale również **system wykonujący je jako automat
> stanów**. Tmux, workery i scheduler są częścią infrastruktury dowodu, natomiast o przejściu między
> etapami zawsze decydują zapisane statystyki i zamrożony kontrakt.

Nie zaczęliśmy od pytania „który model zarobił najwięcej?”. Zaczęliśmy od pytań: czy model potrafi
się uczyć, czy próg decyzji się przenosi, czy cecha rzeczywiście poprawia core, czy przetrwa dane,
które jej nie wybierały, czy przewaga przekracza maksimum powstałe z wielokrotnego szukania, i czy
warto przeznaczyć na nią dodatkowy budżet.

Tmux, workery i scheduler wykonują ten proces automatycznie. Nie decydują jednak, co jest prawdą
naukową. Każda decyzja wynika z kontraktu i zapisanego wyniku.
