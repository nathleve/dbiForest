import numpy as np

from scipy.stats import entropy
from sklearn.cluster import KMeans
import cProfile
from random import shuffle
import memory_profiler
from scipy.stats import chi2
from scipy import stats
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import IsolationForest
import pandas as pd
import random
from math import exp
import math
import copy
from matplotlib.pyplot import boxplot
import importlib.util
from scipy.stats import gaussian_kde
from collections import defaultdict

###### Importance des variables 

def recuperation_features_used_forest(model):
    """
    Étudie l'utilisation des variables au sein d'une forêt d'arbres Isolation/DBiForest.

    Parameters
    ----------
    model : dbiForest object
        Un modèle de dbiForest qui va servir à obtenir les variables utilisées.

    Returns
    -------
    Used : pandas.DataFrame
        Tableau contenant les variables utilisées, la profondeur et la qualité associée
        pour chaque noeud de chaque arbre de la forêt.
    """
    rows = []
    for tree in model.trees:
        rows = recuperation_features_used_tree(tree, rows)
        
    Used = pd.DataFrame(rows)
    return Used


def recuperation_features_used_tree(tree, rows):
    """
    Fonction récursive pour parcourir un arbre et enregistrer les variables utilisées.

    Parameters
    ----------
    tree : dbiTree object
        Un arbre du modèle dbiForest.
    rows : list of dict
        Accumulateur pour stocker les informations de chaque noeud.

    Returns
    -------
    rows : list of dict
        La liste enrichie avec les informations extraites de cet arbre.
    """

    # Cas terminal : si la branche est un entier (probablement une feuille)
    if isinstance(tree.left, int) or isinstance(tree.right, int):
        return rows

    
    # Descente récursive gauche et droite
    rows = recuperation_features_used_tree(tree.left, rows)
    rows = recuperation_features_used_tree(tree.right, rows)

    # Construction de la ligne pour le noeud actuel
    ligne = {
        "depth": tree.current_depth,
        "retenue": tree.split_feature
    }

    
    for c, split in enumerate(tree.liste_split):
        ligne[f"Combinaison {c}"] = split
        ligne[f"Qualite {c}"] = tree.liste_quality[c]

    rows.append(ligne)
    return rows

def importance_variables(df,profondeur_seuil,max_features = 5):
    """
    Calcule l'importance des variables selon une dbiForest

    Parameters
    ----------
    df : dataframe pandas
        L'ensemble des variables vues et utilisées dans une dbiForest
    profondeur_seuil : int 
        La profondeur maximale des noeud utilisables pour calculer les importances 
    max_features : int 
        Le nombre maximal de combinaisons de variables testées

    Returns
    -------
    Used : pandas.DataFrame
        Tableau contenant les variables utilisées, la profondeur et la qualité associée
        pour chaque noeud de chaque arbre de la forêt.
    """
    combinaison_cols = [col for col in df.columns if "Combinaison" in col]
    # Colonne de la combinaison retenue
    col_ret = "retenue"
    
    # Dictionnaire qui compte le nombre de fois que chaque variable a été vu et retenu
    seen_counter = defaultdict(lambda: defaultdict(int))  # {taille_liste: {élément: count}}
    retained_counter = defaultdict(lambda: defaultdict(int))
    # Boucle sur chaque ligne du dataframe de la forêt
    for _, row in df.iterrows():
        if row['depth'] >= profondeur_seuil:
            continue
        elements_seen = set()
        # On regarde toutes les variables testés en prenant en compte tout les cas de figures de construction de df
        for col in combinaison_cols:
            val = row[col]
            # Si la cellule est NaN → on passe
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            
            # Si c’est déjà une liste ou un tableau → on l’utilise directement
            if isinstance(val, (list, tuple, set)):
                elements = val
            elif hasattr(val, '__iter__') and not isinstance(val, str):
                elements = list(val)
            else:
                # Sinon on considère que c’est une chaîne séparée par des virgules
                elements = str(val).split(",")
            elements_seen.update(elements)
        # On regarde maintenant l'élément retenu 
        val_ret = row[col_ret]
        
        # Cas NaN ou cellule vide
        if val_ret is None or (isinstance(val_ret, float) and pd.isna(val_ret)):
            elements_retained = set()
        
        # Cas liste / tuple / autre itérable non-string
        elif isinstance(val_ret, (list, tuple, set)):
            elements_retained = set(val_ret)
        
        elif hasattr(val_ret, '__iter__') and not isinstance(val_ret, str):
            elements_retained = set(val_ret)
        
        # Cas chaîne séparée par des virgules
        else:
            elements_retained = set(e.strip() for e in str(val_ret).split(",") if e.strip())
        
        # ➤ Taille de la combinaison retenue
        taille_ret = len(elements_retained)
    
        # ➤ Mise à jour des compteurs
        for element in elements_seen:
            seen_counter[taille_ret][element] += 1
        for element in elements_retained:
            retained_counter[taille_ret][element] += 1
    # ➤ Création du DataFrame final
    results = []
    for taille, seen_dict in seen_counter.items():
        for element, nb_seen in seen_dict.items():
            nb_ret = retained_counter[taille].get(element, 0)
            perc = 100 * nb_ret / nb_seen if nb_seen > 0 else 0
            results.append({
                'taille_ret': taille,
                'element': element,
                'vu': nb_seen,
                'retenu': nb_ret,
                'pourcentage': perc
            })

    df_results = pd.DataFrame(results)
        
    df_results['rang'] = df_results.groupby('taille_ret')['pourcentage'] \
                                       .rank(ascending=False, method='average')
        
    # Calcul de la moyenne des rangs pour chaque variable
    df_rang_moyen = df_results.groupby('element')['rang'].mean().reset_index()
    df_rang_moyen = df_rang_moyen.rename(columns={'rang': 'rang_moyen'})
        
    df_rang_moyen = df_rang_moyen.sort_values('rang_moyen')
    return df_rang_moyen
    
### Outils pour la découpe des arbres 
def compute_density(data, bandwidth_multiplier = 1): 
    
    """ 
    Calcule la densité d'un nuage de point
    Parameters
    ----------
    data : array_like
        Le nuage de points sur lequel calculer la densité
    bandwidth_multiplier : float 
        Un réel qui sert à modifier le facteur de la largeur de noyau si besoin
    Outputs 
    ------------
    density : array like 
        L'estimation de densité de chaque point du nuage de point 
    kernel : gaussian_kde object 
        Un estimateur de densité par noyau gaussien entraîné sur le nuage de point  

    Bolean, Bolean Indique l'échec de l'estimation
    """
    
    n,p = data.shape
    if n <= p:
        # Si on ne peut pas calculer la densité alors on arrête l'estimation directement
        return False,False

    try:
        # On essaie de calculer la densité des points
        data_t = data.T

        kernel = gaussian_kde(data_t,bw_method=lambda kde: kde.scotts_factor() * bandwidth_multiplier)

        density = kernel(data_t)
    except Exception as e:
        # Si le calcul échoue alors on indique l'échec de l'estimation 

        return False,False
    return density,kernel
 

    
def choice_density_level(k = 1, 
                         epsilon = None,full_data = None,n_features  = [1,3], n_try = 'sqrt',bandwidth_multiplier = 1):
    """ Choisit un seuil de coupure pour un nuage de points 
        Peut aussi choisir les dimensions sur lesquelles on calcule l'estimation de densité pour couper 
    
    Parameters
    ----------
    k : int 
        Un hyperparamètre pour un tirage de seuil selon la loi uniforme basse 
    epsilon : float 
        Le proportion d'anomalies 
    full_data : array_like 
        Le nuage de points avec toutes les dimensions 
    n_features : list
        Le nombre minimal et maximal de dimensions que l'on  peut considérer pour le calcul d'une densité 
    n_try : int ou str
        Le nombre de tirages que l'on fait pour déterminer la coupure/dimension optimal 
        Si n_try est un float alors il devient un int en fonction du nombre de dimensions 
    bandwidth_multiplier : float 
        Un réel qui sert à modifier la largeur des noyaux si besoin

    Outputs 
    ------------
    seuil_density : float
        Le seuil de coupure des données sur la densité estimée
    left_indices : array_like
        La liste des points à gauche de seuil_density 
    best_split : array_like
        Les dimensions qui ont été retenues pour la coupure 
    density : array_like 
        La densité calculée sur le jeu de données full_data 
    kernel : gaussian_kde object 
        L'estimateur de densité entrainé sur les dimensions retenues de full_data 
    liste_split : list
        Une liste des différentes combinaisons de variables envisagées
    liste_quality : list
        Une liste des scores attribués au différentes combinaisons de variables envisagées pour notre critère de sélection
        
    Bolean, Bolean,Bolean, Bolean.... Indicateur d'échec qui arrête la coupure
    """
    # Intialisation de différentes listes
    
    liste_quality = []
    liste_split = []
    best_split = [False,False]
    # Si n_try est un mot clef alors on le transforme en entier 
    if n_try == 'sqrt': 
        n_try = math.ceil((full_data.shape[1])**(1/2))
    if n_try == 'full':
        n_try = (full_data.shape[1])
    # Si la loi fait partie de celle qui utilise le critère de densité basé sur les incréments 
    liste_kernel = []
    # On initialise le dictionnaire qui va contenir le score de chaque variable 
    best_split = [None,None]
    liste_split = []
    liste_quality = []
    # On choisit le nombre de variables qui seront utilisés aléatoirement
    number_features = random.randint(n_features[0],min(n_features[1],full_data.shape[1]))
    # On essaie plusieurs combinaisons de variable avec la taille retenue 
    for i in range(0,n_try):
        split_feature = np.random.choice(full_data.shape[1],number_features,replace = False)
        split_feature = sorted(split_feature)
        # On calcul la densité sur les variables retenues 
        density,kernel = compute_density(full_data[:, split_feature],bandwidth_multiplier)
        if isinstance(density, bool):
            # Si le calcul de densité échoue sur les dimensions utilisées, on passe à l'itération suivante 
            continue
        # On trie la densité puis on calcule le critère de sélection
        sorted_density = np.sort(density)
        # On utilise les incréments de densité
        diff_density = np.diff(sorted_density)
        if np.sum(diff_density) == 0:
            # Toutes les densités sont identiques le calcul est impossible on passe à l'itération suivante
            continue
        # Normalisation par la somme des incréments
        diff_density = diff_density/np.sum(diff_density)
        # On ne considère que les incréments en dessous de la médiane
        diff_density = diff_density[sorted_density[:-1] < np.median(sorted_density)]
        
        if len(diff_density) <= 0:
            continue
        # On récupère l'indice du plus grand incrément
        max_index = np.argmax(diff_density)
        # On récupère la valeur du plus grand incrément et on l'ajoute dans la liste de suivit 
        quality = abs(diff_density[max_index])
        liste_split.append(split_feature)
        liste_quality.append(quality)
    # Si toutes les itérations ont échouées, on arrête la coupure
    if len(liste_quality) == 0 :
        return False,False,[False,False],False,False,False,False,False 
    # On récupère les variables qui ont amenées au plus grand incrément sur toute les itérations 
    index_max = liste_quality.index(max(liste_quality))
    best_features = liste_split[index_max]
    # On recalcule la densité sur les meilleures features puis on sélectionne le seuil de coupure
    density,kernel = compute_density(full_data[:, best_features],bandwidth_multiplier) 
    if isinstance(density, bool):
        return False,False,[False,False],False,False,False,False,False
    sorted_density = np.sort(density)
    diff_density = np.diff(sorted_density)
    diff_density = diff_density/np.sum(diff_density)
    diff_density = diff_density[sorted_density[:-1] < np.median(sorted_density)]
    max_index = np.argmax(diff_density)
    seuil_density = sorted_density[max_index]
    best_split = [seuil_density,best_features]
    best_split = [seuil_density,best_features]
    

    max_density = density.max()
    min_density = density.min()

    # Quelque test et sécurités supplémentaires pour des cas particuler innatendu mais bloquant pour l'algoritme  
    if seuil_density == min_density or seuil_density == max_density:
        # On ajoute une sécurité pour s'assurer de ne pas bloquer l'algorithme, décalage d'un cran si la coupure est égal au minimum ou maximum' des densités
        vals = np.unique(density)
        if len(vals) < 2:
            second_min = vals[0]
            second_max = vals[-1]
        else:
            second_min = vals[1]
            second_max = vals[-2]
        second_min = vals[1]
        second_max = vals[-2]
        seuil_minimal = min_density + (second_min - min_density)/2
        seuil_maximal = max_density - (max_density - second_max)/2
        seuil_density = max(seuil_minimal,seuil_density)
        seuil_density = min(seuil_maximal,seuil_density)
        # Si le cas est non résolue alors on arrête la découpe pour éviter un arrêt du a une erreur 
        if seuil_density == min_density or seuil_density == max_density:                
            return False,False,[False,False],False,False,False,False,False
    if best_split[1] == None:
        return False,False,[False,False],False,False,False,False,False
    left_indices = density[:,] < seuil_density
    if max_density == min_density:
        seuil_density = False
    if all(x == left_indices[0] for x in left_indices):
        seuil_density = False
    return left_indices,seuil_density,best_split,density,kernel,liste_split,liste_quality,liste_kernel
                    


class IsolationTree_density_online:
    
    """ 
    Cette classe est l'arbre d'isolation basé sur la densité  
    Est utilisée de manière récursive car chaque branche d'un noeud d'un arbre est aussi un arbre 
    functions
    ----------
    init : initialise l'abre d'isolation basé sur la densité 
    
    fit : Entraine l'arbre d'isolation basé sur la densité à partir d'un jeu de données 
    
    predict : Retourne la profondeur associée à chaque point d'un jeu de données à travers l'arbre d'isolation basé sur la densité 

    maj_online : Met à jour l'arbre selon la méthodologie online
   """ 
    def __init__(self,current_depth = 0,dim = [1,1],
                epsilon = None,n_try = 'sqrt',features_used = True, bandwidth_multiplier = 1, nmbr_update = 0,nmbr_visit = 0):
        
        """ Initialisation du dbiTree
        Parameters
        ----------
        current_depth : float
            La profondeur de la racine de l'arbre 
        dim :  list
            Le nombre maximal et minimal de dimensions que l'on peut considérer pour calculer la densité dans un noeud de l'arbre 
        epsilon : float 
            La proportion d'anomalies à fournir au début de l'arbre 
        n_try : int ou str 
            Le nombre de découpes considérées si loi amène à devoir choisir une découpe optimale 
        features_used : array like
            Une liste des features utilisables si précisées par l'utilisateur
        bandwidth_multiplier : float 
        Un réel qui sert à modifier le facteur de la bande passante, attention pas la matrice de covariance
        taille_fixe : Boolean
            Un booléen qui indique si on conserve une taille de combinaison de variables fixes entre les divers n_try
        nmbr_update : Int
            Le nombre de fois que l'arbre a été mis à jour 
        nmbr_visit : Int
            Le nombre de fois ou l'arbre a été visité
        Outputs 
        ------------
        dbiTree: dbiTree object 
            Un arbre d'isolation basé sur la densité initialisé mais non entrainé 
        """ 
        self.seuil = None
        self.current_depth = current_depth 
        self.split_feature = None
        self.dim = dim 
        self.epsilon = epsilon
        self.n_try = n_try 
        self.features_used = features_used
        self.bandwidth_multiplier = bandwidth_multiplier
        self.train_idx = None
        self.test_idx = None
        self.date = 0
        self.nmbr_update = nmbr_update
        self.nmbr_visit = nmbr_visit
    def fit(self, X):
        
        """ Initialisation d'un dbiTree
        Parameters
        ----------
        X : array like
            Les points d'entrainement
        Outputs 
        ------------
        dbiTree : dbiTree object 
            Un arbre d'isolation basé sur la densité entrainé 
        """
    
        # Restriction des variables utilisées si demandé par l'utilisateur
        if self.features_used is not True:
            X = X[:,list(self.features_used)]

        num_samples, num_features = X.shape
        self.n = num_samples
        if len(X.shape) != 2:
            # La forme de X n'est pas adaptée il y a un problème dans les données d'entrée ou le code
            print("erreur shape X")
            self.left = self.current_depth
            self.right = self.current_depth
            return self.current_depth
        
        
        # On choisit la découpe + les informations sur la sélection de la découpe
        left_indices,seuil,need,density, kernel,liste_split,liste_quality,liste_kernel = choice_density_level(k=1,
                                                            n_features = self.dim,epsilon = self.epsilon,
                                                           full_data = X,n_try = self.n_try,
                                                            bandwidth_multiplier = self.bandwidth_multiplier)
        
        self.liste_split = liste_split
        self.liste_quality = liste_quality 
        self.liste_kernel = liste_kernel
        self.seuil = seuil 
        # Si on a des Booléens de retournés alors la découpe s'arrête ici, le noeud est une feuille de l'arbre
        if need[1] is False:
            # L'indicateur qu'un noeud est une feuille est que sa droite et sa gauche sont des entiers
            self.left = self.current_depth
            self.right = self.current_depth
            return self.current_depth
        if need[1] is not False:
            self.split_feature = need[1]
        # On vérfie que l'on a bien le droit de découper le noeud selon notre critère d'arrêt et pour éviter un arrêt forcé
        if isinstance(density, bool):
            self.left = self.current_depth
            self.right = self.current_depth
            return self.current_depth
        if seuil == False: 
            self.left = self.current_depth
            self.right = self.current_depth
            return self.current_depth

        self.kernel = kernel
        # Création par récurrence de l'arbre de droite et de gauche 
        right_indices = ~left_indices
        
        self.left = IsolationTree_density_online(self.current_depth + 1, 
                                           dim = self.dim,
                                            epsilon=self.epsilon,n_try = self.n_try,
                                          features_used = True,bandwidth_multiplier = self.bandwidth_multiplier,
                                                nmbr_update = self.nmbr_update)
        self.right = IsolationTree_density_online(self.current_depth + 1, 
                                            dim = self.dim,
                                            epsilon = self.epsilon,n_try = self.n_try,
                                           features_used = True,bandwidth_multiplier = self.bandwidth_multiplier,
                                            nmbr_update = self.nmbr_update)

        self.left.fit(X[left_indices])
        self.right.fit(X[right_indices])
    def update_date(self):
        # Petite fonction récursive qui permet de suivre quand est ce que les noeuds ont été mis à jour
        self.date += 1 
        if isinstance(self.left,int) == True or isinstance(self.right,int) == True:
            return True
        if isinstance(self.left,int) == False and isinstance(self.right,int) == False:
            self.left.update_date()
            self.right.update_date()

    def liste_numero(self, data_numero = None):
        # Petite fonction récursive qui récupère une numérotation unique des noeuds
        if data_numero == None: 
            data_numero = []
        data_numero.append(self.indice)
        if isinstance(self.left, int) == False : 
            data_numero = data_numero + self.left.liste_numero()
        if isinstance(self.right, int) == False : 
            data_numero = data_numero + self.right.liste_numero()
        return data_numero


    def numerotation(self,indice_max):
        # Petite fonction récursive qui numérote de manière unique les noeuds
        if hasattr(self, "indice") == False:    
            self.indice = indice_max
        current_depth = self.current_depth
        indice_max_left = 0 
        indice_max_right = 0 
        if isinstance(self.left, int) == False : 
            indice_max = self.left.numerotation(indice_max + 1)
        if isinstance(self.right, int) == False : 
            indice_max = self.right.numerotation(indice_max + 1)
        return indice_max


    def maj_online(self,new_X,
                  seuil_fusion = 20):
        """ Mise à jour d'un dbiTree
        Parameters
        ----------
        new_X : array like
            Les nouveaux points utilisés pour mettre à jour l'arbre 
        seuil_fusion : int
        Indique la taille maximale des noeuds qui peuvent être mis à jour
        
        Outputs 
        ------------
        dbiT : dbiTree object 
            Un arbre d'isolation basé sur la densité mis à jour avec new_X 
        """
        
        current_update = self.nmbr_update
        if len(new_X) == 0:
            return True
        # On utilise tout les points qui ont parcourut le noeud
        if not hasattr(self, "database"):
            self.database = new_X.copy()
            self.date_database = np.full((len(new_X),), self.date)
        else : 
            self.database = np.vstack((self.database, new_X))
            self.date_database = np.concatenate((
                    self.date_database,
                    np.full((len(new_X),), self.date)
                ))
        if (isinstance(self.left, int) ==  1 and isinstance(self.right, int) ==  1):
            # On est dans une feuille, on arrête la mise à jour
            return True
        # Si on se trouve dans un noeud non terminal alors on continue la descente online à gauche et à droite 
        density = self.kernel.evaluate(np.transpose(new_X[:,self.split_feature]))
        left_indices = density[:,] < self.seuil
            
        right_indices = ~left_indices
                
        data_node = self.kernel.dataset
        
        self.nmbr_visit +=  len(new_X)
        # On vérifie si le noeud est valide pour être mis à jour
        if data_node.shape[1] <= seuil_fusion:
            new_data_node = self.database
            full_data_new = np.vstack((new_data_node,new_X))
            # On vérifie si on a assez de point accumulés pour mettre à jour
            if full_data_new.shape[0] >= data_node.shape[1]: 
                # Si oui et oui alors on refait un sous arbre à partir du noeud 
                left_indices,seuil,need,density, kernel,liste_split,liste_quality,liste_kernel = choice_density_level(k=1,
                                                            n_features = self.dim,epsilon = self.epsilon,
                                                           full_data = full_data_new,n_try = self.n_try,
                                                            bandwidth_multiplier = self.bandwidth_multiplier)
                if need[1] == False:
                    return True
                
                # On efface les points accumulé sur l'ancien noeud 
                del self.database

                self.kernel = kernel
                self.split_feature = need[1]
                self.liste_quality = liste_quality
                self.liste_split = liste_split
                self.liste_kernel = liste_kernel
                self.seuil = seuil 
                
                new_X_left = full_data_new[left_indices]
                new_X_right = full_data_new[~left_indices]

                self.nmbr_update = current_update + 1
                # On créé le nouveau sous arbre de manière récursive
                self.left = IsolationTree_density_online(self.current_depth + 1, 
                                                       dim = self.dim,
                                                     epsilon=self.epsilon,n_try = self.n_try,
                                        features_used = True,bandwidth_multiplier = self.bandwidth_multiplier,
                                                        nmbr_update = current_update + 1)
                self.left.fit(new_X_left)
            
                self.right = IsolationTree_density_online(self.current_depth + 1, 
                                                    dim = self.dim,
                                                    epsilon = self.epsilon,n_try = self.n_try,
                                                   features_used = True,bandwidth_multiplier = self.bandwidth_multiplier,
                                                         nmbr_update = current_update + 1)
                self.right.fit(new_X_right)
                # Après création et entrainement du sous arbre, arrêt de la mise à jour 
                return True 
        # On continue les mise à jour de manière récursive
        self.left.maj_online(new_X[left_indices],seuil_fusion=seuil_fusion)
        self.right.maj_online(new_X[right_indices],seuil_fusion=seuil_fusion)
        return True
        
    def predict(self, X, array_profondeurs, array_indices=None, return_index=False):
        """ Prediction à partir d'un dbiTree de manière récursive
        Parameters
        ----------
        X : array like
            les points sur lequelle on applique le modèle
        array_profondeur : array like
            Sert au suivit de la profondeur
        array_indices : array like
            Sert au suivit des points 
        return index: Boolean
            Indique que l'on veut retourner le chemin des données en plus des profondeurs
        
        Outputs 
        ------------
        result : array like
            Les profondeurs des observations de X
        """
        if array_indices is None:
            array_indices = np.full(X.shape[0], -1)
    
        # Restriction aux features utilisées si précisées par l'utilisateur
        if self.features_used is not True:
            if X.shape[1] > len(self.features_used):
                X = X[:, list(self.features_used)]
    
        # Feuille
        if isinstance(self.left, int) and isinstance(self.right, int):
            if return_index:
                array_indices[:] = self.indice
                return array_profondeurs, array_indices
            return array_profondeurs

        density = self.kernel.evaluate(np.transpose(X[:, self.split_feature]))
        left_mask = density < self.seuil
        # Traitement des points à gauche
        if not isinstance(self.left, int):
            
            out_left = self.left.predict(
                X[left_mask],
                array_profondeurs[left_mask],
                array_indices[left_mask],
                return_index
            )
            if return_index:
                left_depth, left_idx = out_left
            else:
                left_depth = out_left
            left_depth = left_depth + 1
        else:
            left_depth = array_profondeurs[left_mask]
            left_idx = array_indices[left_mask]
    
        # Traitement des points à droite
        if not isinstance(self.right, int):
            out_right = self.right.predict(
                X[~left_mask],
                array_profondeurs[~left_mask],
                array_indices[~left_mask],
                return_index
            )
            if return_index:
                right_depth, right_idx = out_right
            else:
                right_depth = out_right
            right_depth = right_depth + 1
        else:
            right_depth = array_profondeurs[~left_mask]
            right_idx = array_indices[~left_mask]
    
        # Assemblage des profondeurs obtenues récursivement
        result = np.empty_like(array_profondeurs)
        result[left_mask] = left_depth
        result[~left_mask] = right_depth

        if return_index:
            indices_final = np.empty_like(array_indices)
            indices_final[left_mask] = left_idx
            indices_final[~left_mask] = right_idx
            return result, indices_final
    
        return result

class IsolationForest_density_online:
    
    """
        Cette classe est la forêt d'isolation basée sur la densité  
    
        Elle constitué d'une multitude d'arbres d'isolations basés sur la densité 
        functions
        ----------
        init : initialise la forêt 

        fit : Entraine les arbres de la forêt 

        predict : Detecte les anomalies sur un jeu de test

        generate_profondeur :  Récupère la distribution de profondeur sur l'ensemble des arbres pour un nuage de point 

        get_scores_normalized: Retourne les scores d'anormalité des points 

        numérotation : Donne un identifiant unique à chaque noeud de chaque arbre de la forêt

        liste_numéro : Sert à récupérer l'intégralité des identifiants des noeuds de la forêt 

    """ 
    def __init__(self, n_trees=100, subsample_size=0.5,dim = [1,1],epsilon = None,n_try = 'sqrt',bandwidth_multiplier = 1,progression = True):
        
        """ Initialisation de la dbiForest
   
        Parameters
        ----------
        n_trees : int
            Le nombre d'arbre  
        subsample_size :  int
            La taille des sous échantillons pour entrainer chaque arbre si pas d'Out of Bag 
        dim : list 
            Le nombre minimal et maximal de dimension à considérer pour chaque estimation de densité 
        epsilon : float
            Le ratio d'anomalie, epsilon = 0.1 veut dire qu'il y a 10% d'anomalie 
        n_try : int ou str
            Le nombre de découpe considéré si il y a un choix de découpe optimale 
        bandwidth_multiplier : float 
        Un réel qui sert à modifier le facteur de la bande passante, attention pas la matrice de covariance
        Progression: Boolean
            Indique si l'on veut afficher ou en est la forêt dans l'entrainement
        Outputs 
        ------------
        dbiForest : dbiForest object 
            Une forêt d'isolation basé sur la densité initialisé mais pas entrainé
        """
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.dim = dim 
        self.epsilon = epsilon
        self.oob_indices = [] 
        self.n_try = n_try 

        self.features_selected = []
        self.bandwidth_multiplier = bandwidth_multiplier
        self.progression = progression  
        self.nmbr_maj = 0 
        self.window_size = 0
    def numerotation(self):
        """ Cette fonction sert à assurer une numérotation des noeuds cohérente unique et avec laquelle un numéro ne peut jamais se répéter au fur 
        et à mesure des mise à jour online 
        
        Parameters
        ----------
        Outputs 
        ------------
        dbiForest : dbiForest object 
            Une forêt d'isolation avec une numérotation de noeud complète 
        """
        self.liste_indice_max = [0 for _ in range(self.n_trees)] 
        for compteur_trees, tree in enumerate(self.trees):
            self.liste_indice_max[compteur_trees] = tree.numerotation(self.liste_indice_max[compteur_trees])
    def liste_numero(self):
        """ Cette fonction sert à récupérer l'intégralité des numéros présent dans chaque arbre de la forêt 
        
        Parameters
        ----------
        Outputs 
        ------------
        liste_numero : dbiForest object 
            Une forêt d'isolation avec une numérotation de noeud complète 
        """
        
        data_numero = pd.DataFrame()
    
        for j, tree in enumerate(self.trees):
            numeros = tree.liste_numero()   # liste / array / Series
            data_numero[j] = pd.Series(numeros)
    
        return data_numero

    def fit(self, X):
        
        """ Entrainement de la dbiForest
        
        Parameters
        ----------
        X : array_like
            Le nuage de point que l'on utilise pour l'entrainement 

        Outputs 
        ------------
        dbiForest : dbiForest object 
            Une forêt d'isolation basé sur la densité entrainée 
        """

        self.n_point = len(X)
        num_samples = X.shape[0]
        self.trees = []
        self.features_used = [True] * self.n_trees
        points_per_tree = []
        for _ in range(self.n_trees):
            if self.progression:
                print(f"Arbre numéro [{_}] ")
            # On tire le sous échantillon et on entraine l'arbre avec
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(indices, train_size=self.subsample_size)
            X_train, X_valid = X[train_idx], X[test_idx]
                
            subsample = X_train
            points_per_tree.append(indices)
            tree = IsolationTree_density_online(dim = self.dim, epsilon = self.epsilon,n_try = self.n_try,
                                        bandwidth_multiplier = self.bandwidth_multiplier)
            self.points_per_tree = points_per_tree
            tree.train_idx = train_idx
            tree.test_idx = test_idx
            tree.root = tree.fit(subsample)
            self.trees.append(tree)
        # On numérote les noeuds des arbres 
        self.numerotation()
        # On construit le seuil qui sépare les anomalies des points normaux

        depth,index = self.generate_profondeurs(X,use_oob = True,return_index = True)

        invalid_mask = depth.isna().all(axis=1)     # True = à supprimer
        valid_mask = ~invalid_mask                  # True = à garder
        
        
        depth_clean = depth.loc[valid_mask]
        
        moy = depth_clean.mean(skipna=True)
        median = depth_clean.median(skipna=True)
        maximum = depth_clean.max(skipna=True)
        self.threshold_mean = moy.dropna().quantile(self.epsilon)
        self.depth = depth_clean

 
   
    def get_scores_normalized(self,X):
        
        """ Cette fonction permet de retourner des scores d'anormalité de manière similaire à l'iForest de base

            Parameters
            ----------
            X : array like
                Le nuage de point sur lequel on veut détecter les anomalies 
            Outputs 
            ------------
            score : array like 
               Les scores de chaque point basé sur leurs profondeurs dans l'arbre  """
        # Le score est une formule basé sur les profondeurs de X
        profondeur = self.generate_profondeurs(X)
        profondeur = profondeur.mean()
        H = 2*(np.log(len(X)) + 0.5572)*(len(X) - 1)  - 2 * (len(X) - 1)/len(X) 
        profondeur = -profondeur/H
        profondeur = profondeur.astype(float)
        score = np.exp(profondeur)
        return score 
        
    def maj_online(self,new_X, seuil_fusion = 20,
                  window_size = [100]):
        """
        Fonction qui permet de mettre à jour la dbiForest de manière online.
    
        Parameters
        ----------
        old_X : array-like
            Le nuage de points sur lequel on va mettre à jour la forêt
        old_chemin : list
            Indicateur du chemin prit par les ancien points dans les arbres :
            0 si nouveaux points, sinon déjà vus.
        window_size : list
            La liste des windows size a considérer, attention, la première doit être celle qui sert à calculer le threshold par défaut 
        Outputs
        -------
        dbiForest : dbiForest object
            Forêt d'isolation basée sur la densité mise à jour.
        """
        print("in maj_online")
        self.nmbr_maj = self.nmbr_maj + 1
        full_new_chemin = []
        i = 0
        profondeurs = pd.DataFrame(index=range(self.n_trees), columns=range(len(new_X)))
        all_index = pd.DataFrame(index=range(self.n_trees), columns=range(len(new_X)))
        for compteur_trees, tree in enumerate(self.trees):

            # Echantillonage online des données
            mask = np.random.rand(len(new_X)) < 0.75

            X_new_bootstrap = new_X[~mask]
            sample = new_X[mask]
            tree.update_date()
            tree.maj_online(sample, seuil_fusion = seuil_fusion)
            tree.numerotation(0)
            
            depth_array = np.zeros(len(X_new_bootstrap)) 
            depth,index = tree.predict(X_new_bootstrap,depth_array,return_index = True)
            col_idx = np.where(~mask)[0]
            profondeurs.loc[i, col_idx] = depth
            all_index.loc[i, col_idx] = index
            i = i + 1
        max_size = max(window_size)
        if self.depth.shape[1] > max_size:
            # Randomisation des profondeurs pour récupérer autant de points que window_size
            cols = np.random.choice(self.depth.columns, size=max_size, replace=False)
            self.depth = self.depth.loc[:, cols]
            
        if self.depth.shape[1] == max_size:
            # Fenêtre glissante : concaténer puis ne garder que les dernières colonnes
            self.depth = pd.concat([self.depth, profondeurs], axis=1)
            self.depth = self.depth.iloc[:, -max_size:]
            
        thresholds = []
        self.window_size = window_size
        for ws in window_size:

            sub_depth = self.depth.iloc[:, -ws:]       # derniers ws points

            serie = sub_depth.mean(axis=0)             # taille ws

            thresh = serie.quantile(self.epsilon)      # scalaire
            thresholds.append(thresh)

        self.threshold_mean = thresholds[0]
        self.thresholds = thresholds
        self.n_point = self.n_point + len(new_X)

    def predict(self, X, return_index = False):
        
        """ Fonction de prédiction de la dbiForest 

            Parameters
            ----------
            X : array like
                Le nuage de point sur lequel on veut détecter les anomalies 
            
            Outputs 
            ------------
            prediction : array like 
               Les prédictions de la forêt 1 sont les anomalies et 0 les données normales 

        """
        # On commence par récupérer la distribution de profondeur des points 

        if return_index == True:
            depth,indices = self.generate_profondeurs(X =  X,return_index = True)
        if return_index == False:
            depth = self.generate_profondeurs(X = X ,return_index = False)
        y_pred = []

        stats_df = depth.mean()
        y_pred.append(np.where(stats_df >= self.threshold_mean, 0, 1))
        if return_index == False:
            if len(y_pred) == 1:
                return y_pred[0]
            else:
                return y_pred

        if return_index == True:
            print(len(y_pred))
            print(y_pred)
            if len(y_pred) == 1:
                return y_pred[0],indices
            else:
                return y_pred,indices
    def generate_profondeurs(self,X,limit = None,use_oob = False,return_index = False):
        
        """ 
        
        Fonction de récupération de profondeur simple de la dbiForest
   
        Parameters
        ----------
        X : array like
            Le nuage de point sur lequel on récupère la distribution de profondeur
        limit : list
            Indique les arbres sur lesquels on veut calculer les profondeurs si on veut restreindre la forêt 
        use_oob : Boolean
            Si use_oob == True alors la fonction va utiliser sur le X uniquement les éléments qui correspondent aux indices des échantillons oob contenus dans
            les arbres 
        return_index : Boolean
            Permet de demander en plus des profondeurs la trajectoire dans les arbres des points 
        Outputs 
        ------------
        profondeurs : array like 
            Une distribution de profondeurs
            
        """
        profondeurs = pd.DataFrame(index=range(self.n_trees), columns=range(len(X)))
        index = pd.DataFrame(index=range(self.n_trees), columns=range(len(X)))
        if limit is None:
            limit = list(range(self.n_trees))
        
        for i, tree in enumerate(self.trees):
        
            if i not in limit:
                continue
            # Cette condition permet de calculer les profondeurs out of bag, use_oob = True n'a aucun sens si X n'est pas le dataset d'entrainement 
            if use_oob:
                X_used = X[tree.test_idx]
                col_idx = tree.test_idx
            else:
                X_used = X
                col_idx = range(len(X))
        
            depth_array = np.zeros(len(X_used))

            if return_index == True:
                
                depth_test, index_test = tree.predict(
                    X_used, depth_array, return_index=True
                )
                index.loc[i, col_idx] = index_test


            if return_index == False :
                depth_test = tree.predict(
                    X_used, depth_array, return_index=False
                )
            # On retourne toujours au moins la profondeur 
            profondeurs.loc[i, col_idx] = depth_test
        if return_index:
            return profondeurs, index
        return profondeurs

                                    
                                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
