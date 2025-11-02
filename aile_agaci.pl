ebeveyn(ilhami,turker).
ebeveyn(halise,türker).
ebeveyn(ilhami,ayşe).
ebeveyn(halise,ayşe).
ebeveyn(ilhami,alperen).
ebeveyn(halise,alperen).
ebeveyn(mehmet,ilhami).
ebeveyn(mehmet,memduh).
ebeveyn(mehmet,nurgül).
ebeveyn(remziye,ilhami).
ebeveyn(remziye,memduh).
ebeveyn(remziye,nurgül).
ebeveyn(alaettin,halise).
ebeveyn(alaettin,fatma).
ebeveyn(fitnat,halise).
ebeveyn(alaettin,fatih).
ebeveyn(fitnat,fatih).
ebeveyn(fitnat,ayşe).

erkek(ilhami).
erkek(turker).
erkek(alperen).
erkek(mehmet).
erkek(memduh).
erkek(alaettin).
erkek(fatih).

kadin(nurgül).
kadin(remziye).
kadin(fatma).
kadin(ayşe).
kadin(halise).
kadin(fitnat).

anne(X, Y) :- 
    ebeveyn(X, Y), 
    kadin(X).
baba(X, Y) :- 
    ebeveyn(X, Y), 
    erkek(X).
kardes(X, Y) :- 
    ebeveyn(P, X), 
    ebeveyn(P, Y), 
    X \== Y.
kizkardes(X, Y) :- 
    kardes(X, Y), 
    kadin(X).
dede(X, Y) :- 
    baba(X, P), 
    ebeveyn(P, Y).
buyukanne(X, Y) :- 
    anne(X, P), 
    ebeveyn(P, Y).
hala(X, Y) :- 
    baba(Baba, Y), 
    kizkardes(X, Baba).
teyze(X, Y) :- 
    anne(Anne, Y), 
    kizkardes(X, Anne).