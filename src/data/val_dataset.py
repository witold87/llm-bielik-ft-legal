import pandas as pd

validation_questions = [
    'Kiedy jestem konsumentem?',
    'Czy jestem konsumentem, gdy kupuję towar od osoby prywatnej?',
    'Co oznacza wyraźna zgoda na dodatkową płatność za wykonanie umowy?',
    'Czy przedsiębiorca może dowolnie ustalać koszt połączenia ze swoją infolinią?',
    'Czy przy zakupie biletu do teatru obowiązują przepisy ustawy o prawach konsumenta?',
    'Kiedy jest niedozwolona odsprzedaż konsumentom przez przedsiębiorcę biletów na różnego rodzaju imprezy kulturalne lub sportowe?',
    'Kiedy podwójna jakość produktów jest niezgodna z prawem?',
    'Czy do umów o najem i rezerwację miejsc parkingowych zawieranych przez internet stosuje się ustawę o prawach konsumenta?',
    'Czy sprzedawca może przenieść na konsumenta odpowiedzialność za przesyłkę wskazując, np. w treści aukcji, sformułowanie „List ekonomiczny jest nierejestrowany. Nie ponosimy odpowiedzialności za dostarczenie takiej przesyłki”? ',
    'Czy konsument może żądać od sprzedawcy potwierdzenia treści umowy? ',
    'Czy dozwolone jest zawarcie przez przedsiębiorcę z konsumentem umowy pożyczki podczas pokazu lub wycieczki?'
]

groundtruth_answers = [
    'Konsument to osoba fizyczna dokonująca z przedsiębiorcą czynności prawnej niezwiązanej bezpośrednio z jej działalnością gospodarczą lub zawodową. Uznanie za konsumenta ma istotne znaczenie prawne – od tego często zależy jakie przepisy zostaną zastosowane do oceny całej transakcji. W wielu przypadkach sytuacja prawna konsumenta jest z góry wzmacniana przez przepisy.',
    'Nie jesteś konsumentem, gdy kupujesz towar od osoby prywatnej (fizycznej).Konsumentem jest się tylko w sytuacji zakupu od przedsiębiorcy – także od osoby fizycznej prowadzącej działalność gospodarczą. Tylko w tym przypadku kupujący może korzystać z uprawnień przysługujących konsumentom – w pozostałych sytuacjach prawa te nie przysługują.',
    'Wyraźną zgodę wyraża sam konsument. Nie można zakładać jej przez zastosowanie domyślnych opcji, które trzeba dopiero odrzucić. Zgody nie można domniemywać na podstawie tego, że konsument nie wyraził sprzeciwu. Konsument ma prawo do zwrotu płatności dodatkowej, na którą nie wyraził wyraźnej zgody.',
    'Nie, przedsiębiorca nie może ustalać tego w sposób dowolny. Jeżeli wskazuje numer telefonu do kontaktu z nim w sprawie zawartej umowy, opłata dla konsumenta za połączenie z tym numerem nie może być wyższa niż opłata za zwykłe połączenie telefoniczne, zgodnie z pakietem taryfowym dostawcy usług, z którego korzysta konsument.',
    'Tak, przepisy ustawy o prawach konsumenta stosuje się do umów zawieranych przez tego typu placówki. Teatr jest przedsiębiorcą w rozumieniu przepisów Kodeksu cywilnego, niezależnie od tego, czy jest finansowany ze środków publicznych, czy też prywatnych. Uwaga! W odniesieniu do umów zawartych poza lokalem przedsiębiorstwa lub na odległość i związanych z wydarzeniami rozrywkowymi lub kulturalnymi, konsumentowi nie przysługuje prawo odstąpienia od umowy.',
    'Niedozwolona jest odsprzedaż konsumentom biletów na wszelkiego rodzaju imprezy kulturalne lub sportowe, jeżeli przedsiębiorca nabył je z wykorzystaniem oprogramowania pozwalającego mu obchodzić środki techniczne lub przekraczać limity techniczne nałożone na pierwotnego sprzedawcę, aby obejść ograniczenia nałożone na liczbę biletów, które dana osoba może kupić, lub inne zasady dotyczące zakupu biletów.',
    'Niezgodne z prawem jest wprowadzenie na polski rynek towaru, który jest oferowany jako identyczny z towarem wprowadzonym na rynek innego kraju UE, a w rzeczywistości istotnie różni się on składem lub właściwościami. Nie dotyczy to sytuacji, gdy taki stan rzeczy wynika z obiektywnych i uzasadnionych czynników.',
    'Tak, w przypadku tego typu umów stosuje się ustawę o prawach konsumenta. Zgodnie z treścią preambuły dyrektywy w sprawie praw konsumentów (wdrożonej do polskiego prawa ustawą o prawach konsumenta), jej przepisami powinny być objęte m.in. umowy dotyczące najmu pomieszczeń do celów innych niż mieszkalne. Przy czym ustawy o prawach konsumenta nie stosuje się m.in. do umów dotyczących najmu pomieszczeń do celów mieszkalnych.',
    'Przedsiębiorca nie może przenieść na konsumenta odpowiedzialności za przesyłkę. Wynika to z dwóch kwestii: 1) zawierając umowę na odległość sprzedawca odpowiada za towar do momentu wydania go „do ręki” konsumentowi. Nie ma znaczenia czy korzystał z usług listonosza czy kuriera. Sprzedawca odpowiada przed konsumentem za nieprawidłowe wywiązanie się z umowy zarówno w przypadku zagubienia, jak i uszkodzenia przesyłki w trakcie transportu. Konsument odpowiada za transport tylko wtedy, gdy to on sam wybierze pośrednika – dostawcę towaru. 2) wskazanie przez przedsiębiorcę w sposób wyraźny i jednoznaczny w umowie – np. w treści aukcji – że nie ponosi on odpowiedzialności za przesyłkę jest nieważne z mocy prawa. Jeżeli sprzedawca zastosuje takie sformułowanie, a towar nie dojdzie do konsumenta, to sprzedawca będzie musiał wysłać towar jeszcze raz – na własny koszt, bez dodatkowej dopłaty. Jest to zatem ryzyko przedsiębiorcy – to on musi udowodnić, że doręczył produkt, a nie konsument, że mu doręczono.',
    'Jeżeli przedsiębiorca proponuje konsumentowi zawarcie umowy przez telefon, ma obowiązek potwierdzić treść proponowanej umowy utrwaloną na papierze lub innym trwałym nośniku. Oświadczenie konsumenta o zawarciu umowy jest skuteczne, jeżeli zostało utrwalone na papierze lub innym trwałym nośniku po otrzymaniu potwierdzenia od przedsiębiorcy. Przedsiębiorca ma obowiązek przekazać konsumentowi potwierdzenie zawarcia umowy na odległość na trwałym nośniku w rozsądnym czasie po jej zawarciu, najpóźniej w chwili dostarczenia rzeczy lub przed rozpoczęciem świadczenia usługi.',
    'Nie, umowa dotycząca usług finansowych nie może być zawarta podczas pokazu lub wycieczki. Zakaz ten obejmuje także zawarcie umowy dotyczącej usług finansowych związanej bezpośrednio z ofertą złożoną podczas pokazu lub wycieczki w celu realizacji umowy sprzedaży. W tej sytuacji umowa będzie nieważna.Możliwe jest jednak zawarcie umowy dotyczącej usług finansowych podczas pokazów w miejscu zamieszkania lub zwykłego pobytu konsumenta, organizowanych na jego wyraźne zaproszenie.'
]


def get_validation_dataset():
    return pd.DataFrame({'question': validation_questions, 'groundtruth': groundtruth_answers})