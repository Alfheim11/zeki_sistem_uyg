belirti_var(ates, VarMi) :-
    write('Atesin var mı? (evet/hayir): '), read(Cevap),
    (
        Cevap = evet ->
            write('Kac derece?: '), read(Derece),
            (Derece >= 38 -> VarMi = 1 ; VarMi = 0)
        ;
        VarMi = 0
    ).
belirti_var(oksuruk, VarMi) :-
    write('Öksürüğün var mi? (evet/hayir): '), read(Cevap),
    (Cevap = evet -> VarMi = 1 ; VarMi = 0).
belirti_var(nefes_darligi, VarMi) :-
    write('Nefes darlığın var mı? (evet/hayir): '), read(Cevap),
    (Cevap = evet -> VarMi = 1 ; VarMi = 0).
belirti_var(tat_kaybi, VarMi) :-
    write('Tat ve ya koku kaybın var mı? (evet/hayir): '), read(Cevap),
    (Cevap = evet -> VarMi = 1 ; VarMi = 0).
korona_tahmini :-
    belirti_var(ates, A),
    belirti_var(oksuruk, B),
    belirti_var(nefes_darligi, C),
    belirti_var(tat_kaybi, D),
Toplam is A + B + C + D,
    (
        Toplam >= 3 ->
            write('Test yaptırmalısınız korona olabilirsiniz.')
        ;
            write('Diğer belirtiler de olursa sağlık ocağına başvurun şuanlık bir sorununuz yok.')
    ).