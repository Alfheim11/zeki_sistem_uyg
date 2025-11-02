ogrenci(turker,90,80,70,90,2).
ogrenci(kerem,80,70,60,56,3).
ogrenci(nihat,50,30,4,73,8).
ogrenci(gürkan,23,45,55,90,1).
ogrenci(eren,60,80,100,90,0).
ogrenci(arda,15,2,56,40,3).

ogrenci_durumu(Isim, kaldi(devamsizliktan)) :-
    ogrenci(Isim, _, _, _, _, Devamsizlik),
    Devamsizlik > 5, !.

ogrenci_durumu(Isim, Durum) :-
    ogrenci(Isim, Arasinav, Odev, Final, Butunleme, _),     
    VizeNotu is (Arasinav * 0.5) + (Odev * 0.5),
    (   Butunleme > Final
    ->  GecerliFinalNotu = Butunleme  
    ;   GecerliFinalNotu = Final      
    ),
    Ortalama is (VizeNotu * 0.4) + (GecerliFinalNotu * 0.6),
    (   Ortalama > 50
    ->  Durum = gecti(Ortalama)
    ;   Durum = kaldi(Ortalama)  
    ).