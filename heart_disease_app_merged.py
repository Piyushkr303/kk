
# ===== base.py =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def splitter(data, y_var='DISEASE'):

   # Splitting the data into dependent & independent variables -
    X = data.drop(columns=y_var, axis=1).values
    y = data[y_var].values

    return X, y


from sklearn.preprocessing import StandardScaler

def standardizer(X_train, X_test):
    
    # Standardizing the data -
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_scaled = np.concatenate([X_train_scaled, X_test_scaled], axis=0)
    return X_scaled, X_train_scaled, X_test_scaled


def standardize(X):
    
    # Standardizing the data -
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    return X_scaled


def model_train(model_obj, X_train, y_train, **kwargs): 

    model_obj.fit(X_train, y_train, **kwargs)
    return model_obj


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def model_eval(model_obj, X_train, X_test, y_train, y_test):

    y_pred_test = model_obj.predict(X_test)
    y_pred_test_proba = model_obj.predict_proba(X_test)[:, 1]
    
    print("Train accuracy: {:.2f}%".format(accuracy_score(y_train, model_obj.predict(X_train)) * 100))
    print("Test accuracy: {:.2f}%".format(accuracy_score(y_test, model_obj.predict(X_test)) * 100))
    print("F1 Score: {:.2f}".format(f1_score(y_test, y_pred_test)))
    print("Precision: {:.2f}".format(precision_score(y_test, y_pred_test)))
    print("Recall: {:.2f}".format(recall_score(y_test, y_pred_test)))
    print("ROC AUC Score: {:.2f}".format(roc_auc_score(y_test, y_pred_test_proba)))
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    tpr = tp/(tp+fn)
    tnr = tn/(fp+tn)
    fpr = fp/(tp+fn)
    fnr = fn/(fp+tn)
    
    print("Type 1 Error: {:.2f}".format(fpr))
    print("Type 2 Error: {:.2f}".format(fnr))
    print("Sensitivity: {:.2f}".format(tpr))
    print("Specificity: {:.2f}\n".format(1-fpr))
    
    return y_pred_test, y_pred_test_proba
    
    
def show_pred(y_pred_test, y_pred_test_proba):
    pred = pd.DataFrame({'Probability': y_pred_test_proba,
                          'Class': y_pred_test})
    print("\n", pred)

    
from sklearn.model_selection import KFold, cross_val_score
  
def cross_val(model_obj, X, y, scoring='f1'):
    kfold = KFold(n_splits=5)
    
    score = np.mean(cross_val_score(model_obj, X, y, cv=kfold, scoring=scoring, n_jobs=-1)) 
    print("Cross Validation Score: {:.2f}".format(score))
    
    
from sklearn.metrics import roc_auc_score, roc_curve

def roc_auc_curve_plot(model_obj, X_test, y_test): 
    logit_roc_auc = roc_auc_score(y_test, model_obj.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model_obj.predict_proba(X_test)[:,1])
    
    plt.figure()
    plt.plot(fpr, tpr, label='(area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


from sklearn.metrics import precision_recall_curve

def precision_recall_curve_plot(model_obj, X_test, y_test):
    y_pred_proba = model_obj.predict_proba(X_test)[:,1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    threshold_boundary = thresholds.shape[0]
    
    # plot precision
    plt.plot(thresholds, precisions[0:threshold_boundary], label='precision')
    # plot recall
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recalls')
    
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    
    plt.xlabel('Threshold Value'); plt.ylabel('Precision and Recall Value')
    plt.legend(); plt.grid()
    plt.show()




# ===== ann.py =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

import warnings
warnings.simplefilter('ignore')


# Loading the train & test data -
train = pd.read_csv(r'C:\Users\piyus\OneDrive\Desktop\Heart-Disease-Pred-main\Heart-Disease-Pred-main\train2.csv')
test = pd.read_csv(r'C:\Users\piyus\OneDrive\Desktop\Heart-Disease-Pred-main\Heart-Disease-Pred-main\test2.csv')


# Splitting the data into independent & dependent variables -
X_train, y_train =  base.splitter(train, y_var='DISEASE')
X_test, y_test =  base.splitter(test, y_var='DISEASE')


# Standardizing the data -
X_scaled, X_train_scaled, X_test_scaled = base.standardizer(X_train, X_test)


model = keras.Sequential([
                          keras.layers.Dense(units=128, input_shape=(13,), activation='relu', kernel_regularizer=regularizers.l2(2.0)),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(3.0)),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(3.0)),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(3.0)),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(3.0))
                         ])


adam=keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])


print(model.summary())


es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-3, patience=10, mode='max', verbose=0)
mc = tf.keras.callbacks.ModelCheckpoint(filepath='model.h5', save_best_only=True, save_weights_only=True)


hist = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), 
                 epochs=100, batch_size=32, callbacks=[es, mc], verbose=1)


_, train_acc = model.evaluate(X_train_scaled, y_train, batch_size=32, verbose=0)
_, test_acc = model.evaluate(X_test_scaled, y_test, batch_size=32, verbose=0)

print('Train Accuracy: {:.3f}'.format(train_acc))
print('Test Accuracy: {:.3f}'.format(test_acc))


y_pred_proba = model.predict(X_test_scaled, batch_size=32, verbose=0)

threshold = 0.60
y_pred_class = np.where(y_pred_proba > threshold, 1, 0)


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


plt.figure(figsize=(10, 5))
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='test')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


from sklearn.metrics import roc_auc_score, roc_curve

logit_roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_class)

plt.figure()
plt.plot(fpr, tpr, label='(area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import precision_recall_curve
    
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

threshold_boundary = thresholds.shape[0]

# plot precision
plt.plot(thresholds, precisions[0:threshold_boundary], label='precision')
# plot recall
plt.plot(thresholds, recalls[0:threshold_boundary], label='recalls')

start, end = plt.xlim()
plt.xticks(np.round(np.arange(start, end, 0.1), 2))

plt.xlabel('Threshold Value'); plt.ylabel('Precision and Recall Value')
plt.legend(); plt.grid()
plt.show()


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_class)

plt.figure(figsize = (10, 7))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class))


# Saving the model -
model_json = model.to_json()
with open(r'C:\Users\piyus\OneDrive\Desktop\Heart-Disease-Pred-main\Heart-Disease-Pred-main\ann.py', 'w') as json_file:
    json_file.write(model_json)
    
# Serialize weights to HDF5 -
model.save_weights(r'C:\Users\piyus\OneDrive\Desktop\Heart-Disease-Pred-main\Heart-Disease-Pred-main\ann.py')

# ===== heart_disease_rbm_classifier.py =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
import streamlit as st
import joblib
import warnings
warnings.simplefilter('ignore')

# Function to split data into X and y components
def splitter(data, y_var):
    """Split data into X and y components"""
    X = data.drop(y_var, axis=1)
    y = data[y_var]
    return X, y

# Function to standardize features
def standardizer(X_train, X_test):
    """Standardize features using training data statistics"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

def normalize_for_rbm(X_train_scaled, X_test_scaled):
    """Normalize data to [0,1] for RBM"""
    min_max_scaler = MinMaxScaler()
    X_train_normalized = min_max_scaler.fit_transform(X_train_scaled)
    X_test_normalized = min_max_scaler.transform(X_test_scaled)
    return min_max_scaler, X_train_normalized, X_test_normalized

print("Loading and preparing data...")
# Loading the train & test data
train = pd.read_csv('train2.csv')
test = pd.read_csv('test2.csv')

# Splitting the data into independent & dependent variables
X_train, y_train = splitter(train, y_var='DISEASE') 
X_test, y_test = splitter(test, y_var='DISEASE')

# Standardizing the data
std_scaler, X_train_scaled, X_test_scaled = standardizer(X_train, X_test)

# Normalize data to range [0,1] for RBM
min_max_scaler, X_train_normalized, X_test_normalized = normalize_for_rbm(X_train_scaled, X_test_scaled)

# Create and train an RBM + LogisticRegression pipeline
print("Creating Boltzmann Machine pipeline...")
rbm = BernoulliRBM(
    n_components=50,  # Number of hidden units
    learning_rate=0.01,
    n_iter=20,  # Number of iterations
    verbose=True,
    random_state=42
)

# Create pipeline: RBM for feature extraction + LogisticRegression for classification
rbm_classifier = Pipeline(steps=[
    ('rbm', rbm),
    ('logistic', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000))
])

print("Training the Boltzmann Machine classifier...")
# Train the model
rbm_classifier.fit(X_train_normalized, y_train)

# Model evaluation
train_acc = rbm_classifier.score(X_train_normalized, y_train)
test_acc = rbm_classifier.score(X_test_normalized, y_test)

print('Train Accuracy: {:.3f}'.format(train_acc))
print('Test Accuracy: {:.3f}'.format(test_acc))

# Make predictions
y_pred_proba = rbm_classifier.predict_proba(X_test_normalized)[:, 1]

threshold = 0.60
y_pred_class = np.where(y_pred_proba > threshold, 1, 0)

# Visualize the learned RBM features
plt.figure(figsize=(10, 5))
for i, comp in enumerate(rbm.components_):
    if i < 10:  # Display only a subset of components
        plt.subplot(2, 5, i + 1)
        plt.imshow(comp.reshape((1, -1)), cmap=plt.cm.RdBu_r, 
                   interpolation='nearest', aspect='auto')
        plt.xticks(())
        plt.yticks(())
        plt.title(f'Component {i+1}')
plt.suptitle('RBM Feature Visualization (First 10 components)', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('rbm_features.png')  # Save for Streamlit
plt.close()

# ROC curve and AUC score
logit_roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label=f'AUC = {logit_roc_auc:.2f}')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve.png')  # Save for Streamlit
plt.close()

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

threshold_boundary = thresholds.shape[0]
plt.figure(figsize=(10, 5))
plt.plot(thresholds, precisions[:threshold_boundary], label='Precision')
plt.plot(thresholds, recalls[:threshold_boundary], label='Recall')
plt.xlabel('Threshold Value')
plt.ylabel('Precision and Recall Value')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig('precision_recall.png')  # Save for Streamlit
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.savefig('confusion_matrix.png')  # Save for Streamlit
plt.close()

# Classification Report
report = classification_report(y_test, y_pred_class)
print(report)

# Save the model, scalers and components
joblib.dump(rbm_classifier, 'rbm_classifier_model.pkl')
joblib.dump(std_scaler, 'std_scaler.pkl')
joblib.dump(min_max_scaler, 'min_max_scaler.pkl')

# Function to visualize RBM weights for Streamlit
def visualize_weights(rbm_model):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    feature_names = ["Feature " + str(i+1) for i in range(5)]
    
    for i, ax in enumerate(axes.flatten()):
        if i < min(5, rbm_model.components_.shape[0]):
            ax.bar(range(rbm_model.components_.shape[1]), rbm_model.components_[i])
            ax.set_title(f"{feature_names[i]}")
            ax.set_xticks([])
    
    plt.tight_layout()
    return fig

# Streamlit App
def rbm_app():
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
            color: #1E88E5;
        }
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .metric-header {
            color: #1E88E5;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f8ff;
            margin: 20px 0;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<p class="main-header">Heart Disease Prediction with Boltzmann Machine</p>', unsafe_allow_html=True)
    
    @st.cache_resource
    def load_model_and_scalers():
        model = joblib.load('rbm_classifier_model.pkl')
        std_scaler = joblib.load('std_scaler.pkl')
        mm_scaler = joblib.load('min_max_scaler.pkl')
        return model, std_scaler, mm_scaler
    
    # Load model and scalers
    try:
        loaded_model, std_scaler, mm_scaler = load_model_and_scalers()
        model_loaded = True
    except:
        st.warning("Model files not found. Please run the training code first.")
        model_loaded = False
    
    # Create tabs for input and analysis
    tab1, tab2, tab3 = st.tabs(["📝 Input Parameters", "📊 Model Analysis", "🧠 RBM Features"])
    
    with tab1:
        # Create two columns with better spacing
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### Patient Metrics")
            AGE = st.number_input("Age", min_value=0, max_value=120, step=1, 
                                help="Enter patient's age")
            RESTING_BP = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, step=1,
                                       help="Enter resting blood pressure in mm Hg")
            SERUM_CHOLESTROL = st.number_input("Serum Cholesterol", min_value=0, max_value=1000, step=1,
                                             help="Enter serum cholesterol level in mg/dl")
            TRI_GLYCERIDE = st.number_input("Triglycerides", min_value=0, max_value=1000, step=1,
                                          help="Enter triglyceride level in mg/dl")
            LDL = st.number_input("LDL", min_value=0, max_value=300, step=1,
                                help="Enter LDL cholesterol level in mg/dl")
            HDL = st.number_input("HDL", min_value=0, max_value=100, step=1,
                                help="Enter HDL cholesterol level in mg/dl")
            FBS = st.number_input("Fasting Blood Sugar", min_value=0, max_value=500, step=1,
                                help="Enter fasting blood sugar level in mg/dl")
           
        with col2:
            st.markdown("### Clinical Parameters")
            GENDER = st.selectbox('Gender', 
                                options=[0, 1], 
                                format_func=lambda x: "Female" if x == 0 else "Male",
                                help="Select patient's gender")
            
            CHEST_PAIN = st.selectbox('Chest Pain', 
                                    options=[0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes",
                                    help="Presence of chest pain")
            
            RESTING_ECG = st.selectbox('Resting ECG', 
                                     options=[0, 1],
                                     format_func=lambda x: "Normal" if x == 0 else "Abnormal",
                                     help="Resting electrocardiographic results")
            
            TMT = st.selectbox('TMT (Treadmill Test)', 
                             options=[0, 1],
                             format_func=lambda x: "Normal" if x == 0 else "Abnormal",
                             help="Treadmill test results")
            
            ECHO = st.number_input("Echo", min_value=0, max_value=100, step=1,
                                 help="Echocardiogram value")
            
            MAX_HEART_RATE = st.number_input("Maximum Heart Rate", 
                                           min_value=0, max_value=250, step=1,
                                           help="Maximum heart rate achieved")
        
        # Collect all inputs
        encoded_results = [AGE, GENDER, CHEST_PAIN, RESTING_BP, SERUM_CHOLESTROL, 
                         TRI_GLYCERIDE, LDL, HDL, FBS, RESTING_ECG, MAX_HEART_RATE, 
                         ECHO, TMT]
        
        # Add a predict button
        if st.button('Predict', type='primary', use_container_width=True):
            if model_loaded:
                # Show a spinner while predicting
                with st.spinner('Analyzing with Boltzmann Machine...'):
                    sample = np.array(encoded_results).reshape(1, -1)
                    # Scale the input using both scalers in sequence
                    sample_scaled = std_scaler.transform(sample)
                    sample_normalized = mm_scaler.transform(sample_scaled)
                    prediction = loaded_model.predict_proba(sample_normalized)[0, 1]
                
                # Display prediction in a nice box
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h2>Prediction Result</h2>
                        <h1 style="font-size: 48px; color: #1E88E5;">{prediction:.2%}</h1>
                        <p>Probability of Heart Disease</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.error("Please train the model first")
    
    with tab2:
        if model_loaded:
            st.markdown("""
            ### Model Evaluation Metrics
            
            Explore the various metrics used to evaluate the Boltzmann Machine model's performance:
            """)
            
            metrics = st.radio(
                "Select Metric to View:",
                ["ROC-AUC Curve", "Precision-Recall", "Confusion Matrix", "Classification Report"],
                horizontal=True
            )
            
            # Show metrics
            if metrics == "ROC-AUC Curve":
                with st.expander("📈 ROC-AUC Curve", expanded=True):
                    st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
                    st.image('roc_curve.png')
                    st.markdown(f"**AUC Score:** {logit_roc_auc:.4f}")
                    
            elif metrics == "Precision-Recall":
                with st.expander("📊 Precision-Recall Curve", expanded=True):
                    st.markdown("#### Precision-Recall Curve")
                    st.image('precision_recall.png')
                    
            elif metrics == "Confusion Matrix":
                with st.expander("🔢 Confusion Matrix", expanded=True):
                    st.markdown("#### Model Confusion Matrix")
                    st.image('confusion_matrix.png')
                    
            elif metrics == "Classification Report":
                with st.expander("📝 Classification Report", expanded=True):
                    st.markdown("#### Detailed Classification Metrics")
                    st.code(report)
            
            # Add explanation of metrics
            with st.expander("📚 Understanding the Metrics"):
                st.markdown("""
                #### Detailed Explanation of Evaluation Metrics
                
                1. **ROC-AUC Curve**
                - Plots true positive rate vs false positive rate
                - Higher AUC indicates better model discrimination
                - Perfect classifier would have AUC = 1.0
                
                2. **Precision-Recall Plot**
                - Shows trade-off between precision and recall
                - Helps in choosing optimal threshold for classification
                - Important for imbalanced datasets
                
                3. **Confusion Matrix**
                - Shows true positives, false positives, true negatives, and false negatives
                - Helps understand model's classification performance
                - Useful for identifying specific types of errors
                
                4. **Classification Report**
                - Provides precision, recall, f1-score, and support for each class
                - Gives a comprehensive overview of model performance
                - Helps identify class imbalance issues
                """)
        else:
            st.warning("Please train the model first to view model evaluation metrics.")
    
    with tab3:
        if model_loaded:
            st.markdown("### RBM Feature Visualization")
            st.markdown("""
            This tab shows the features learned by the Restricted Boltzmann Machine. 
            RBMs learn latent representations of the input data which capture underlying patterns.
            """)
            
            st.image('rbm_features.png')
            
            with st.expander("🔍 Understanding RBM Features"):
                st.markdown("""
                #### How Restricted Boltzmann Machines Work
                
                1. **Unsupervised Learning**: RBMs learn patterns in data without labels
                
                2. **Hidden Units**: Each component (hidden unit) captures a different pattern in the data
                
                3. **Feature Extraction**: The visualization shows the connection weights between input features and hidden units
                
                4. **Interpretation**: 
                   - Red areas indicate positive weights (features that activate the hidden unit)
                   - Blue areas indicate negative weights (features that inhibit the hidden unit)
                   - The stronger the color, the stronger the relationship
                
                5. **Benefits for Classification**:
                   - RBMs can discover complex non-linear patterns
                   - The learned features often make classification tasks easier
                   - They can help detect important relationships between input variables
                """)
        else:
            st.warning("Please train the model first to visualize RBM features.")

if __name__ == "__main__":
    rbm_app()


# ===== heart_disease_voting_classifier.py =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
import streamlit as st
import joblib
import warnings
warnings.simplefilter('ignore')

# This assumes you have the base.py module with splitter and standardizer functions
# I'll reimplement these if they don't exist
def splitter(data, y_var):
    """Split data into X and y components"""
    X = data.drop(y_var, axis=1)
    y = data[y_var]
    return X, y

def standardizer(X_train, X_test):
    """Standardize features using training data statistics"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

# Loading the train & test data
train = pd.read_csv('train2.csv')
test = pd.read_csv('test2.csv')

# Splitting the data into independent & dependent variables
X_train, y_train = splitter(train, y_var='DISEASE') 
X_test, y_test = splitter(test, y_var='DISEASE')

# Standardizing the data
scaler, X_train_scaled, X_test_scaled = standardizer(X_train, X_test)

# Define individual classifiers
log_clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
svc_clf = SVC(kernel='rbf', C=1, probability=True, random_state=42)

# Create the voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('rf', rf_clf),
        ('svc', svc_clf)
    ],
    voting='soft'  # Use 'soft' voting to get probability outputs
)

# Train the model
print("Training the voting classifier...")
voting_clf.fit(X_train_scaled, y_train)

# Evaluate on train and test data
train_acc = voting_clf.score(X_train_scaled, y_train)
test_acc = voting_clf.score(X_test_scaled, y_test)

print('Train Accuracy: {:.3f}'.format(train_acc))
print('Test Accuracy: {:.3f}'.format(test_acc))

# Make predictions
y_pred_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]

threshold = 0.60
y_pred_class = np.where(y_pred_proba > threshold, 1, 0)

# ROC curve and AUC score
logit_roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label='AUC = %0.2f' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve.png')  # Save for Streamlit
plt.close()

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

threshold_boundary = thresholds.shape[0]
plt.figure(figsize=(10, 5))
plt.plot(thresholds, precisions[0:threshold_boundary], label='Precision')
plt.plot(thresholds, recalls[0:threshold_boundary], label='Recall')
plt.xlabel('Threshold Value')
plt.ylabel('Precision and Recall Value')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig('precision_recall.png')  # Save for Streamlit
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.savefig('confusion_matrix.png')  # Save for Streamlit
plt.close()

# Classification Report
report = classification_report(y_test, y_pred_class)
print(report)

# Save the model and scaler
joblib.dump(voting_clf, 'voting_classifier_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Streamlit App
def voting_clf_app():
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
            color: #1E88E5;
        }
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .metric-header {
            color: #1E88E5;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f8ff;
            margin: 20px 0;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<p class="main-header">Heart Disease Prediction Model</p>', unsafe_allow_html=True)
    
    @st.cache_resource
    def load_model_and_scaler():
        model = joblib.load('voting_classifier_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    
    # Load model and scaler
    try:
        loaded_model, loaded_scaler = load_model_and_scaler()
        model_loaded = True
    except:
        st.warning("Model files not found. Please run the training code first.")
        model_loaded = False
    
    # Create tabs for input and analysis
    tab1, tab2 = st.tabs(["📝 Input Parameters", "📊 Model Analysis"])
    
    with tab1:
        # Create two columns with better spacing
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### Patient Metrics")
            AGE = st.number_input("Age", min_value=0, max_value=120, step=1, 
                                help="Enter patient's age")
            RESTING_BP = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, step=1,
                                       help="Enter resting blood pressure in mm Hg")
            SERUM_CHOLESTROL = st.number_input("Serum Cholesterol", min_value=0, max_value=1000, step=1,
                                             help="Enter serum cholesterol level in mg/dl")
            TRI_GLYCERIDE = st.number_input("Triglycerides", min_value=0, max_value=1000, step=1,
                                          help="Enter triglyceride level in mg/dl")
            LDL = st.number_input("LDL", min_value=0, max_value=300, step=1,
                                help="Enter LDL cholesterol level in mg/dl")
            HDL = st.number_input("HDL", min_value=0, max_value=100, step=1,
                                help="Enter HDL cholesterol level in mg/dl")
            FBS = st.number_input("Fasting Blood Sugar", min_value=0, max_value=500, step=1,
                                help="Enter fasting blood sugar level in mg/dl")
           
        with col2:
            st.markdown("### Clinical Parameters")
            GENDER = st.selectbox('Gender', 
                                options=[0, 1], 
                                format_func=lambda x: "Female" if x == 0 else "Male",
                                help="Select patient's gender")
            
            CHEST_PAIN = st.selectbox('Chest Pain', 
                                    options=[0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes",
                                    help="Presence of chest pain")
            
            RESTING_ECG = st.selectbox('Resting ECG', 
                                     options=[0, 1],
                                     format_func=lambda x: "Normal" if x == 0 else "Abnormal",
                                     help="Resting electrocardiographic results")
            
            TMT = st.selectbox('TMT (Treadmill Test)', 
                             options=[0, 1],
                             format_func=lambda x: "Normal" if x == 0 else "Abnormal",
                             help="Treadmill test results")
            
            ECHO = st.number_input("Echo", min_value=0, max_value=100, step=1,
                                 help="Echocardiogram value")
            
            MAX_HEART_RATE = st.number_input("Maximum Heart Rate", 
                                           min_value=0, max_value=250, step=1,
                                           help="Maximum heart rate achieved")
        
        # Collect all inputs
        encoded_results = [AGE, GENDER, CHEST_PAIN, RESTING_BP, SERUM_CHOLESTROL, 
                         TRI_GLYCERIDE, LDL, HDL, FBS, RESTING_ECG, MAX_HEART_RATE, 
                         ECHO, TMT]
        
        # Add a predict button
        if st.button('Predict', type='primary', use_container_width=True):
            if model_loaded:
                # Show a spinner while predicting
                with st.spinner('Analyzing...'):
                    sample = np.array(encoded_results).reshape(1, -1)
                    # Scale the input using the same scaler used during training
                    sample_scaled = loaded_scaler.transform(sample)
                    prediction = loaded_model.predict_proba(sample_scaled)[0, 1]
                
                # Display prediction in a nice box
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h2>Prediction Result</h2>
                        <h1 style="font-size: 48px; color: #1E88E5;">{prediction:.2%}</h1>
                        <p>Probability of Heart Disease</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.error("Please train the model first")
    
    with tab2:
        if model_loaded:
            st.markdown("""
            ### Model Evaluation Metrics
            
            Explore the various metrics used to evaluate the model's performance:
            """)
            
            metrics = st.radio(
                "Select Metric to View:",
                ["ROC-AUC Curve", "Precision-Recall", "Confusion Matrix", "Classification Report"],
                horizontal=True
            )
            
            # Show metrics
            if metrics == "ROC-AUC Curve":
                with st.expander("📈 ROC-AUC Curve", expanded=True):
                    st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
                    st.image('roc_curve.png')
                    st.markdown(f"**AUC Score:** {logit_roc_auc:.4f}")
                    
            elif metrics == "Precision-Recall":
                with st.expander("📊 Precision-Recall Curve", expanded=True):
                    st.markdown("#### Precision-Recall Curve")
                    st.image('precision_recall.png')
                    
            elif metrics == "Confusion Matrix":
                with st.expander("🔢 Confusion Matrix", expanded=True):
                    st.markdown("#### Model Confusion Matrix")
                    st.image('confusion_matrix.png')
                    
            elif metrics == "Classification Report":
                with st.expander("📝 Classification Report", expanded=True):
                    st.markdown("#### Detailed Classification Metrics")
                    st.code(report)
            
            # Add explanation of metrics
            with st.expander("📚 Understanding the Metrics"):
                st.markdown("""
                #### Detailed Explanation of Evaluation Metrics
                
                1. **ROC-AUC Curve**
                - Plots true positive rate vs false positive rate
                - Higher AUC indicates better model discrimination
                - Perfect classifier would have AUC = 1.0
                
                2. **Precision-Recall Plot**
                - Shows trade-off between precision and recall
                - Helps in choosing optimal threshold for classification
                - Important for imbalanced datasets
                
                3. **Confusion Matrix**
                - Shows true positives, false positives, true negatives, and false negatives
                - Helps understand model's classification performance
                - Useful for identifying specific types of errors
                
                4. **Classification Report**
                - Provides precision, recall, f1-score, and support for each class
                - Gives a comprehensive overview of model performance
                - Helps identify class imbalance issues
                """)
        else:
            st.warning("Please train the model first to view model evaluation metrics.")

if __name__ == "__main__":
    voting_clf_app()


# ===== app.py =====

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from base import preprocess_data, evaluate_model
from ann import train_ann, predict_ann
from heart_disease_rbm_classifier import train_rbm, predict_rbm
from heart_disease_voting_classifier import train_voting, predict_voting

st.set_page_config(page_title="Heart Disease Classifier", layout="wide")
st.title("Heart Disease Prediction using Multiple Models")

# Sidebar - Upload Data
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

model_option = st.sidebar.selectbox(
    "Choose a Model",
    ("Artificial Neural Network (ANN)", "Restricted Boltzmann Machine (RBM)", "Voting Classifier")
)

# Placeholder for dataframe
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Dataset")
    st.write(df.head())

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Run selected model
    if st.sidebar.button("Train and Predict"):
        with st.spinner("Training and Predicting..."):
            if model_option == "Artificial Neural Network (ANN)":
                model = train_ann(X_train, y_train)
                y_pred = predict_ann(model, X_test)

            elif model_option == "Restricted Boltzmann Machine (RBM)":
                model = train_rbm(X_train, y_train)
                y_pred = predict_rbm(model, X_test)

            elif model_option == "Voting Classifier":
                model = train_voting(X_train, y_train)
                y_pred = predict_voting(model, X_test)

        # Display Evaluation Metrics
        st.subheader(f"Evaluation Metrics for {model_option}")
        metrics = evaluate_model(y_test, y_pred)
        st.write(metrics)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Bar Plot of Precision, Recall, F1-Score
        st.subheader("Precision, Recall, and F1-Score")
        scores_df = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1-Score"],
            "Score": [metrics['precision'], metrics['recall'], metrics['f1_score']]
        })
        fig2 = plt.figure()
        sns.barplot(x="Metric", y="Score", data=scores_df)
        st.pyplot(fig2)
else:
    st.info("Please upload a dataset to continue.")
