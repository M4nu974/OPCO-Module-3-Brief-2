# M3 - Brief 2 - Étendre un modèle IA à de nouvelles entrées sans perdre les apprentissages précédents

Ce projet illustre le processus de mise à jour d'un modèle de machine learning préexistant en utilisant l'apprentissage par transfert (transfer learning). Un nouveau réseau de neurones est créé pour s'adapter à un jeu de données mis à jour (avec des caractéristiques supplémentaires), et ses poids initiaux sont transférés depuis un ancien modèle pré-entraîné. Le nouveau modèle est ensuite ré-entraîné sur l'ensemble des données complètes.
 
---

## Mise en Route

Suivez ces instructions pour configurer et exécuter le projet localement.

### Prérequis

* Python 3.11 

### Installation

1.  **Clonez le dépôt :**
    ```bash
    git clone <url-depot>
    cd <repertoire-depot>
    ```

2.  **Créez et activez un environnement virtuel (recommandé) :**
    ```bash
    # Pour Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Pour macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Installez les dépendances requises :**
    ```bash
    pip install -r requirements.txt
    ```

---

## Utilisation

Pour lancer le processus d'entraînement du modèle, exécutez le script `main.py` depuis le répertoire racine du projet :

```bash
python main.py
```

Ce script effectuera les actions suivantes :
1.  Charger le jeu de données depuis `dataset/data-all-complete_clean.csv`.
2.  Charger le modèle Keras pré-entraîné (`model_2025.h5`) et le préprocesseur de données (`preprocessor.pkl`).
3.  Prétraiter les données en utilisant les fonctions de `preprocess.py`.
4.  Définir un nouveau modèle `Sequential` Keras (`model_2`) avec une architecture adaptée aux nouvelles données.
5.  Transférer les poids du modèle chargé vers le nouveau modèle pour toutes les couches compatibles.
6.  Compiler et entraîner le nouveau modèle sur les données prétraitées.
7.  Générer et sauvegarder un graphique des courbes de perte (loss) d'entraînement et de validation sous le nom `loss_model2.jpg`.

