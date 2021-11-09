# ANLP A2 Report

The distances were calculated in triplets: two of the sentences were similar and a third was using the same word but in a different context. This was done to emphasize ELMo's capabilities.

## Triplet 1
```
============================================================
i am watching videos on my computer
i am playing games on my laptop
------------------------------------------------------------
computer[0] <-> laptop[1]
Euclidean Distance: 52.403221130371094
   Cosine Distance: 0.3818994164466858
============================================================
```

```
============================================================
i am watching videos on my computer
i am playing soccer with that ball
------------------------------------------------------------
computer[0] <-> ball[2]
Euclidean Distance: 62.01863479614258
   Cosine Distance: 0.5142042934894562
============================================================
```

```
============================================================
i am playing games on my laptop
i am playing soccer with that ball
------------------------------------------------------------
laptop[1] <-> ball[2]
Euclidean Distance: 55.38338851928711
   Cosine Distance: 0.42192572355270386
============================================================
```

## Triplet 2
```
============================================================
i play baseball and golf
he plays a big role at his company
------------------------------------------------------------
play[0] <-> plays[1]
Euclidean Distance: 53.09914779663086
   Cosine Distance: 0.38139408826828003
============================================================
```

```
============================================================
i play baseball and golf
i play soccer in the playground
------------------------------------------------------------
play[0] <-> play[2]
Euclidean Distance: 17.431089401245117
   Cosine Distance: 0.04140061140060425
============================================================
```

```
============================================================
he plays a big role at his company
i play soccer in the playground
------------------------------------------------------------
plays[1] <-> play[2]
Euclidean Distance: 52.39838409423828
   Cosine Distance: 0.378934383392334
============================================================
```

## Triplet 3
```
============================================================
my phone rings when you call me
he is wearing a ring on his finger
------------------------------------------------------------
rings[0] <-> ring[1]
Euclidean Distance: 66.4549560546875
   Cosine Distance: 0.5852328538894653
============================================================
```

```
============================================================
my phone rings when you call me
the lion went through a ring of fire
------------------------------------------------------------
rings[0] <-> ring[2]
Euclidean Distance: 64.56147003173828
   Cosine Distance: 0.5729151368141174
============================================================
```

```
============================================================
he is wearing a ring on his finger
the lion went through a ring of fire
------------------------------------------------------------
ring[1] <-> ring[2]
Euclidean Distance: 31.423025131225586
   Cosine Distance: 0.13287204504013062
============================================================
```
