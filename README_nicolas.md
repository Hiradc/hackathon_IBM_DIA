### I. Processus de création

Pendant ce hackaton, nous avons commencé par déterminer quel résultat nous voulions, qu'est ce que nous voulions développer. Quel était notre objectif, notre rendu. Pour cela, nous avons passé notre mercredi après-midi à brainstormer sur notre concept, à en déterminer les contours, en restant conscient des défis techniques.

Ainsi, nous étions initialement parti sur une extension web, qui agit comme une surcouche sur les site web des plus gros LLMs. Lorsque l'utilisateur se connecte sur le site d'un de ces LLMs, et qu'il interragit avec l'agent conversationnel, l'extension web intègre dans l'interface du site web une nouvelle donnée avec l'information du coup de chaque requête.

Cette solution permettait une rapide prise en main par n'importe qui, car il suffit de télécharger l'extension sur les magasins d'extension web.

Malheureusement, après de nombreux tests mercredi après-midi, nous nous sommes rendus compte que WatsonX bloquait les requêtes API à partir du web. Ainsi, l'extension web était incapable de requêter l'API de WatsonX directement, il fallait passer par un serveur externe, ce qui rendait le projet bien plus complexe, et les coûts bien plus élevés.

Nous avons donc revu notre objectif, et sommes plutôt partis sur une interface web, qui imite les gros sites d'agent conversationnel (comme ChatGPT et Claude), et permet à l'utilisateur d'échanger avec les différents agents conversationnels. Cela fonctionne un peu comme Mamouth, avec un portail et une même interface qui permet de parler avec différents modèles. Cette interface entièrement développée par nous nous permet d'être bien plus libre techniquement.

Mais l'objectif n'est pas seulement de copier le fonctionnement de Mamouth, mais de rajouter notre propre valeur ajoutée, avec l'ajout de l'empreinte carbone de chaque requête.

### II. Difficultés techniques

Initialement, nous avions prévu d'héberger notre modèle d'IA dans WatsonX, mais du au fait que pour déployer un modèle dans WatsonX, il faut un espace de stockage, et donc un abonnement payant. Nous n'avons donc pas pu mettre l'IA dans WatsonX. Ainsi, l'IA de prédiction de consommation électrique, ainsi que l'interface web sont hébergées en local

### III. Empreinte carbone de chaque requête

Une fois que notre IA de prédiction de consommation électrique d'une requête a prédit la consommation de la requête, on mets cette consommation dans la formule ci dessous pour obtenir l'empreinte carbone de cette requête.

$$
\text{Empreinte\_carbone\_de\_la\_requête} =
\text{Energie\_consommée\_par\_la\_requête} \times 
\text{Empreinte\_carbone\_d'un\_kWh} +
\frac{\text{Empreinte\_carbone\_de\_l\_entrainement\_du\_modèle}}{\text{Nombre\_moyen\_de\_requêtes}}
$$

Dans notre formule, nous avons choisi de prendre en compte le coût de l'entrainement du modèle. En effet, la majorité de l'empreinte carbone d'un modèle vient de son entrainement. Nos calculs étaient donc vains si nous ne prenions pas en compte ce paramètre.
