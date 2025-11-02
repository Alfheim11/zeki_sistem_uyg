and_gate(1,0,0).
and_gate(1,1,1).
and_gate(0,1,0).
and_gate(0,0,0).

or_gate(1,0,1).
or_gate(1,1,1).
or_gate(0,1,1).
or_gate(0,0,0).

not_gate(1,0).
not_gate(0,1).

devre(Giris1, Giris2, Giris3, Cikis) :-
    and_gate(Giris1, Giris2, T1),
    or_gate(Giris2, Giris3, T2),
    and_gate(T1, T2, T3),
    not_gate(T3, Cikis).


