# Movie Recommender System using Restricted Boltzmann Machines (RBM)

A sophisticated movie recommendation system implemented using Restricted Boltzmann Machines (RBM) for collaborative filtering. This project demonstrates how RBMs can learn latent factors from user-movie interactions to provide personalized movie recommendations.

## 🎯 Overview

This implementation uses RBMs, which are generative stochastic neural networks that can learn probability distributions over their inputs. The system converts explicit movie ratings (1-5 stars) to binary implicit feedback and learns hidden representations that capture user preferences and movie characteristics.

## 🔑 Key Features

- **Collaborative Filtering**: Learns from user-movie interaction patterns
- **Latent Factor Learning**: Discovers hidden features that influence user preferences
- **Missing Data Handling**: Can work with sparse rating matrices (unrated movies)
- **Binary Feedback Conversion**: Simplifies explicit ratings to implicit feedback
- **Contrastive Divergence Training**: Efficient learning algorithm for RBMs

## 📊 Dataset

The system uses the **MovieLens** dataset:
- **MovieLens 100K**: Training and test splits (80/20)
- **MovieLens 1M**: Full dataset for comprehensive analysis
- **943 users** and **1,682 movies**
- **1,000,209 total ratings**

### Data Structure

- **Movies**: Movie ID, Title, Genre
- **Users**: User ID, Gender, Age Group, Occupation, Zip Code
- **Ratings**: User ID, Movie ID, Rating (1-5), Timestamp

## 🏗️ Architecture

### RBM Structure
- **Visible Layer**: 1,682 units (movies)
- **Hidden Layer**: 100 units (latent factors)
- **No intra-layer connections** (restricted architecture)

### Training Process
1. **Data Preprocessing**: Convert ratings to user-movie matrix
2. **Binary Conversion**: Ratings ≥3 → 1 (liked), <3 → 0 (not liked)
3. **Contrastive Divergence**: Gibbs sampling for parameter updates
4. **Batch Processing**: Process users in batches of 100

## 🚀 Implementation

### Prerequisites

```bash
pip install pandas numpy torch urllib3
```

### Key Components

#### 1. Data Loading and Preprocessing
```python
# Load datasets
movies = pd.read_csv('ml-1m/ml-1m/movies.dat', sep='::', header=None)
users = pd.read_csv('ml-1m/ml-1m/users.dat', sep='::', header=None)
ratings = pd.read_csv('ml-1m/ml-1m/ratings.dat', sep='::', header=None)

# Convert to user-movie matrix
training_set = convert(training_set)  # 943 x 1682 matrix
```

#### 2. RBM Class Implementation
```python
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)  # Weights
        self.a = torch.randn(1, nh)   # Hidden bias
        self.b = torch.randn(1, nv)   # Visible bias
    
    def sample_h(self, x):  # Forward pass
    def sample_v(self, y):  # Backward pass
    def train(self, v0, vk, ph0, phk):  # Parameter updates
```

#### 3. Training Process
```python
# Contrastive Divergence (CD-10)
for epoch in range(nb_epoch):
    for batch in user_batches:
        # Positive phase
        ph0, _ = rbm.sample_h(v0)
        
        # Gibbs sampling
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
        
        # Negative phase
        phk, _ = rbm.sample_h(vk)
        
        # Update parameters
        rbm.train(v0, vk, ph0, phk)
```

## 📈 Performance

### Model Results
- **Training MAE**: 0.2470
- **Test MAE**: 0.2544
- **Performance**: Excellent (MAE < 0.3)

### Dataset Statistics
- **Total Movies**: 3,883
- **Total Users**: 6,040
- **Total Ratings**: 1,000,209
- **Average Ratings per User**: 165.6
- **Average Ratings per Movie**: 257.6

## 🎯 Key Insights

1. **Latent Factor Discovery**: RBM successfully learns hidden features from user-movie interactions
2. **Binary Simplification**: Converting explicit ratings to binary feedback improves learning
3. **Efficient Training**: Contrastive Divergence enables practical training of RBMs
4. **Sparsity Handling**: Model effectively handles missing data (unrated movies)
5. **Generalization**: Low test MAE indicates good generalization to unseen data

## 🔧 Configuration

### Hyperparameters
- **Hidden Units**: 100 (latent factors)
- **Batch Size**: 100 users
- **Epochs**: 10
- **Gibbs Steps**: 10 (CD-10)
- **Rating Threshold**: 3+ stars = liked

### Rating Conversion
- **Unrated**: -1 (missing data)
- **1-2 stars**: 0 (not liked)
- **3-5 stars**: 1 (liked)

## 🚀 Potential Improvements

1. **Hyperparameter Tuning**
   - Experiment with different numbers of hidden units
   - Try various learning rates
   - Test different rating conversion thresholds

2. **Advanced Techniques**
   - Implement early stopping based on validation loss
   - Add regularization to prevent overfitting
   - Use adaptive learning rates

3. **Model Enhancements**
   - Incorporate movie metadata (genres, release year)
   - Add user demographic information
   - Implement ensemble methods

4. **Evaluation Metrics**
   - Precision@K and Recall@K
   - Normalized Discounted Cumulative Gain (NDCG)
   - Diversity and novelty metrics

## 📁 File Structure

```
section 17 Boltzman Machines/
├── Movie_Recommender_System_RBM_V2.ipynb  # Main implementation
├── Movie_Recommender_System_RBM_Improved.py  # Python version
├── Boltzman Machines.md  # Theory and concepts
└── README.md  # This file
```

## 🎓 Learning Outcomes

This project demonstrates:
- **Deep Learning Fundamentals**: Understanding neural network architectures
- **Collaborative Filtering**: Learning from user-item interactions
- **Probabilistic Models**: Working with RBMs and probability distributions
- **Data Preprocessing**: Handling sparse, categorical data
- **Model Evaluation**: Assessing recommendation system performance

## 🔗 Related Resources

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Restricted Boltzmann Machines Tutorial](https://deeplearning4j.org/docs/latest/deeplearning4j-nn-restrictedboltzmannmachine)
- [Collaborative Filtering with Neural Networks](https://arxiv.org/abs/1409.2944)

## 📝 License

This project is part of the Deep Learning A-Z course and is intended for educational purposes.

---

**Note**: This implementation focuses on the core RBM algorithm for collaborative filtering. For production systems, consider additional factors like scalability, real-time recommendations, and integration with existing platforms. 
