Bonjour, à défaut d'un rapport détaillé, nous vous fournissons ce petit fichier 
vous donnant les informations importantes afin d'utiliser le jeu.


Tout d'abord nous tenons à nous excuser au sujet de la cohérence 
des différentes conditions implémentées (tirée de votre cahier des charges).
Beaucoup de ces conditions ont dû être interprétées (peut-être de manière hasardeuse 
et trop intuitives selon les idées de ce que l'on se fait de la réalité sur le 
fonctionnement d'une centrale nucléaire), ceci est notamment vrai dans la gestion des ouvriers 
et des réparations. A ce propos nous ne sommes pas sûr de la cohérence des unités de mesure
employées pour exprimer les différentes radioactivités.


En ce qui concerne l'utilisation du jeu, vous avez à votre disposition un makefile 
dans le dossier affichage qui vous permettra de compiler l'exécutable
du jeu avec ses trois modes. La partie affichage tient d'un bloc et a très peu de 
commentaires par défaut de temps. Nous espérons que le noms des variables
et les différentes instructions avec les quelques annotations rendront le code lisible.

IMPORTANT : surtout ne modifiez pas les fichiers du dossier SDL2 que nous vous joignons car nous avons dû modifier ces fichiers afin d'y intégrer quelques
fonctionnalités supplémentaires, si vous le souhaitez vous pouvez d'ailleurs ajouter les fonctions d'affichage dans votre librairie pour les années à venir.
Si tel est votre souhait nous nous ferons une joie de mettre ce code plus au propre...


COMMANDES SALLE DE CONTRÔLES : 
Ce jeu se joue principalement à la souris : vous pourrez donc modifier les différents paramètres modifiables de la centrale en cliquant sur les barres. Les zones
qui ne peuvent pas être choisies comme valeurs demandées sont grisées et la valeur demandée actuelle est affichée sous la forme d'un petit trait rouge, la valeur
actuelle de la centrale est, quant à elle symbolisée par un trait blanc (le trait blanc se superpose au trait rouge pour plus de clarté).
touche s : terminer la partie
touche t : switcher d'écran entre salle de contrôle et radioprotection

COMMANDES RADIOPROTECTION :
Les ouvriers peuvent être envoyés lorsque le bouton "GO" associée est de couleur bleue, vous pouvez ajuster le nombre d'ouvriers à l'aide des touches +/-.
Vous avez aussi la possibilité de rapatrier tous les ouvriers d'un organe à l'aide du bouton avec l'icône flèche, à noter que toutes ces actions sont mises
en attente et sont exécutées à la fin de chaque tour.
Remarque : nos quelques expérimentations sur ce jeu montrent que les ouvriers ont tendance à subir une épidémie générale de maladie, est-ce parce que nous sommes
des employeurs irresponsables, envoyant nos ouvriers malgré les radiations, pressions, températures meurtrières... Ou avons-nous tout simplement mal interprêté
votre cahier des charges ?

AFFICHAGE :
	MODE 1 :
		SALLE DE CONTRÔLE :
			Affichage classique : schéma de la centrale, alertes organes en bas à gauche affichées en rouge et actualisées chaque tour. Barres grises
			non modifiables et barres bleues et grises modifiables.
		RADIOPROTECTION :
			Affichage des radioactivités, variables population, ouvriers en haut à droite. Juste en dessous il y a le bouton d'arrêt d'urgence et
			celui permettant l'évacuation de la population. Enfin sur la gauche vous retrouverez une barre représentant respectivement l'état d'un
			organe. A côté des barres des organes réparables se trouvent une zone de boutons afin de gérer les ouvriers sur chaque chantier.
	MODE 2 :
		SALLE DE CONTRÔLE :
			Ajout : affichage des ordres du dispatching national
		RADIOPROTECTION :
			Inchangée.
	MODE 3 :
		SALLE DE CONTRÔLE :
			Affichage juste au dessus des ordre du dispatching des catastrophes qui surviennent lors de la partie d'un joueur malchanceux.
		RADIOPROTECTION :
			Inchangée
	Ecran de fin de jeu :
		Nous affichons ici votre score dispatching et un commentaire à ce propos (si vous n'avez pas joué en mode 1), les états moyens du circuit primaire et secondaire
		avec un commentaire sur l'état de la centrale, la contamination de la population, la production moyenne et un commentaire dessus.

AUTRES : vous pouvez modifier directement la taille de l'écran et la vitesse d'un tour (en ms) dans la fonction main du dossier affichage. 


Nous vous souhaitons un bon jeu.

Willem Szwarc
Louis De Juan
Andréa Gourion