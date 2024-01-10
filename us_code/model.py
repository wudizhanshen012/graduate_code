from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation

my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
from tensorflow.keras import Model
def linear_regression():
    model = LinearRegression()

    return model

def lasso(ps):
    model = LassoCV(alphas=[0.001, 0.0001],
                    cv=ps)

    return model

def ridge(ps):
    model = RidgeCV(alphas=[0.1, 0.01, 0.001, 0.0001],
                    cv=ps)

    return model

def elastic_net(ps):
    model = ElasticNetCV(alphas=[0.001, 0.0001],
                         cv=ps)

    return model

def random_forest(ps):
    param_grid = {'max_features': [3, 5, 10],
                  'max_depth': [1, 2],
                  'n_estimators': [300],
                  }
    model = RandomForestRegressor(n_jobs=10)
    model = GridSearchCV(model, param_grid, cv=ps, n_jobs=3
                         )

    return model

def NN3(inpu, l1, seed):
    x = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
              kernel_regularizer=tf.keras.regularizers.l1(l1)
              )(inpu)
    x = Activation('linear')(x)
    x = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
              kernel_regularizer=tf.keras.regularizers.l1(l1))(x)
    x = Activation('linear')(x)
    x = Dense(8, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
              kernel_regularizer=tf.keras.regularizers.l1(l1))(x)
    x = Activation('linear')(x)
    x = Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
              kernel_regularizer=tf.keras.regularizers.l1(l1))(x)
    model = Model(inpu, x)
    return model
