# Autoencoder-TDLOG
 For TDLog and MOPSI Project at ENPC 
 
Nous avons décidé de faire porter notre projet TDLog sur le sujet que nous traitons en
MOPSI. Le problème qui se pose en modélisation moléculaire est que le temps nécessaire
pour observer de lents changements de conformation du système est généralement plus
grand que l’échelle de temps accessible par la simulation de la dynamique moléculaire.
Cela est dû au fait que le système est bloqué dans des états métastables et empêche ainsi
l’entière exploitation de l’espace configurationnelle. Les quantités thermodynamiques ne
peuvent alors être approchées précisemment. Afin de répondre à ce problème, on peut
s’intéresser aux variables collectives (Collective Variables (CV)), aussi appelées coordonnées de réaction qui fournissent une représentation de petites dimensions du système.
Dans des cas simple, il est facile de définir les variables collectives.
Dans le cas de la dialanine, par exemple, les variables collectives sont les angles dièdres ϕ et ψ. Ces angles caractérisent parfaitement la structure secondaire de la proteine.
Pour des systèmes plus complexes, il est difficile de définir les variables collectives. Des
méthodes de réduction de dimension et de machine learning permettent alors de les estimer. Une méthode d’apprentissage non supervisé, couramment utilisé dans ce domaine,
consiste à identifier ces variables collectives à l’aide d’autoencoders.
Ce sont des réseaux de neurones dont la dimension de la première couche est égale
à la dimension de la dernière avec une couche intermédiaire, appelé bottleneck, dont le
dimension est réduite par rapport aux autres
