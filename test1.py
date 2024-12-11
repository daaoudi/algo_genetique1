import numpy as np

# Exemple de vocabulaire pour les mots positifs et négatifs
vocabulaire = ["excellent", "cher", "mais", "terrible", "fantastique"]
taille_population = 20  # Nombre d'individus dans la population

# Générer une population initiale aléatoire
def initialiser_population(taille_population, vocab_size):
    return np.random.uniform(low=-1, high=1, size=(taille_population, vocab_size))

population = initialiser_population(taille_population, len(vocabulaire))
#print(population)

#fonction de fitness
def fitness(individu, data, labels):
    scores = []
    for phrase, label in zip(data, labels):
        score = sum([individu[vocabulaire.index(mot)] for mot in phrase.split() if mot in vocabulaire])
        pred = 1 if score > 0 else 0  # 1 = Positif, 0 = Négatif
        scores.append(pred == label)  # Prédiction correcte ?
    return sum(scores) / len(scores)  # Précision

#Sélection, croisement et mutation

# Sélectionner les meilleurs individus
def selectionner_meilleurs(population, fitness_scores, n=5):
    indices = np.argsort(fitness_scores)[-n:]  # Les meilleurs n
    return population[indices]

# Croisement
def crossover(parent1, parent2):
    point = np.random.randint(len(parent1))
    enfant = np.concatenate((parent1[:point], parent2[point:]))
    return enfant

# Mutation
def muter(individu, taux_mutation=0.1):
    for i in range(len(individu)):
        if np.random.rand() < taux_mutation:
            individu[i] += np.random.uniform(-0.1, 0.1)
    return individu

#Boucle d’évolution

# Données d'exemple
data = ["excellent produit", "terrible expérience", "cher mais fantastique", "pas cher mais terrible"]
labels = [1, 0, 1, 0]  # 1 = Positif, 0 = Négatif

# Algorithme génétique
n_generations = 50
for generation in range(n_generations):
    # Calcul de la fitness
    fitness_scores = [fitness(individu, data, labels) for individu in population]
    
    # Sélectionner les meilleurs
    meilleurs = selectionner_meilleurs(population, fitness_scores, n=5)
    
    # Générer une nouvelle population
    nouvelle_population = []
    while len(nouvelle_population) < taille_population:
        indices = np.random.choice(len(meilleurs), size=2, replace=False)
        parent1, parent2 = meilleurs[indices[0]], meilleurs[indices[1]]
        enfant = crossover(parent1, parent2)
        enfant = muter(enfant)
        nouvelle_population.append(enfant)
    
    population = np.array(nouvelle_population)
    #print(f"Generation {generation}: Meilleure fitness = {max(fitness_scores):.2f}")



# Obtenir le meilleur individu après évolution
meilleur_individu = population[np.argmax([fitness(individu, data, labels) for individu in population])]

# Fonction pour prédire le sentiment d'une liste de phrases
def predire_sentiments(phrases, individu):
    results = []
    for phrase in phrases:
        score = sum([individu[vocabulaire.index(mot)] for mot in phrase.split() if mot in vocabulaire])
        sentiment = "Positif" if score > 0 else "Négatif"
        results.append((phrase, sentiment))
    return results

# Tester avec une liste de nouvelles phrases
nouveaux_textes = [
    "pas cher mais terrible",
    "excellent produit et fantastique service",
    "cher mais acceptable",
    "terrible qualité et mauvais service"
]

resultats = predire_sentiments(nouveaux_textes, meilleur_individu)

# Afficher les résultats
for texte, sentiment in resultats:
    print(f"Phrase : {texte} | Sentiment : {sentiment}")
