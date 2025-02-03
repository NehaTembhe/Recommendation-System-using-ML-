from flask import Flask, request, render_template, flash, redirect, url_for,  session
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os
import logging
from flask_migrate import Migrate
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.decomposition import TruncatedSVD
import numpy as np
from datetime import datetime


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load files
base_dir = os.path.abspath(os.path.dirname(__file__))

# Use relative path for training data to make it portable
train_data_path = os.path.join(base_dir, "models/flipkart_com-ecommerce_sample.csv")

try:
    train_data = pd.read_csv(train_data_path)
    app.logger.debug(f"Train data loaded successfully with {train_data.shape[0]} rows and {train_data.shape[1]} columns.")
except OSError as e:
    app.logger.error(f"Error loading training data: {e}")
    flash(f"Error loading training data: {e}", 'error')
    train_data = pd.DataFrame()  # Set to an empty DataFrame to avoid further errors


# Debugging paths
app.logger.debug(f"BASE_DIR: {base_dir}")
app.logger.debug(f"Training Data Path: {train_data_path}")


# Database configuration
app.secret_key = "your_secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:@localhost:3307/ecomm"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Replace with your actual database URL
DATABASE_URL = "mysql+pymysql://root:123456789@localhost:3307/ecomm"

# Create the engine
engine = create_engine(DATABASE_URL)

# SQLAlchemy session
Session = sessionmaker(bind=engine)
db_session = Session()  # Renamed to avoid conflicts


# Initialize Flask-Migrate
migrate = Migrate(app, db)

train_data['product_rating'] = pd.to_numeric(train_data['product_rating'], errors='coerce')

class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)  # Store the original password


# Define your model class for the 'signin' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)
    product_id = db.Column(db.Integer, nullable=False)  # Remove the ForeignKey reference
    action_type = db.Column(db.String(50))  # e.g., "click", "purchase", "search"
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp)



#Funtions
def truncate_filter(value, length=28):
    """Truncate a string to a specified length."""
    if len(value) > length:
        return value[:length] + "..."
    return value


# Register the custom filter
app.jinja_env.filters['truncate'] = truncate_filter


# User activity tracker
user_activity_data = []


def insert_user_activity(user_id, product_id, action_type):
    """Insert user activity into the database."""
    print(f"DEBUG: product_id type: {type(product_id)}, value: {repr(product_id)}")

    if not product_id or isinstance(product_id, (str, int)) and str(product_id).strip() == "":
        print("Error: product_id is empty or None. User activity not recorded.")
        return

    product_id = str(product_id).strip()  # Convert to string and trim spaces

    # Ensure product_id does not contain invalid characters
    if any(c in product_id for c in ['\n', '\r', '\t', '\x00']):
        print(f"Error: Invalid characters in product_id: {repr(product_id)}")
        return

    try:
        user_activity = UserActivity(
            user_id=user_id,
            product_id=product_id,
            action_type=action_type,
            timestamp=datetime.now()
        )
        db.session.add(user_activity)
        db.session.commit()
        print(f"User activity recorded: {action_type} for product ID '{product_id}'")
    except Exception as e:
        db.session.rollback()
        print(f"Database error: {e}")

def get_top_rated_products(n=28):
    """ Returns top N products with a rating of 5 and specific columns from the first 2000 rows """
    # Limit the data to first 2000 rows
    limited_train_data = train_data.head(10000)

    # Convert product_rating to float (if not already)
    limited_train_data['product_rating'] = limited_train_data['product_rating'].astype(float)

    # Filter products with rating of 5.0
    top_rated_products = limited_train_data[limited_train_data['product_rating'] == 5.0]

    if top_rated_products.empty:
        return pd.DataFrame(columns=['product_name', 'brand', 'image', 'product_rating', 'discounted_price'])

    # Selecting only the required columns
    selected_columns = ['product_name', 'brand', 'image', 'product_rating', 'discounted_price']
    top_rated_products = top_rated_products[selected_columns]

    return top_rated_products.sample(n=min(n, top_rated_products.shape[0])).reset_index(drop=True)


def content_based_recommendations(train_data, item_name, top_n=20):
    # Normalize case and strip extra whitespace
    train_data['product_name'] = train_data['product_name'].str.lower().str.strip()
    item_name = item_name.lower().strip()

    app.logger.debug(f"Searching for product: {item_name}")

    # Verify item exists in the dataset
    matched_products = train_data[train_data['product_name'].str.contains(item_name, case=False, na=False)]
    app.logger.debug(f"Matched products: {matched_products.shape[0]}")

    if matched_products.empty:
            matched_products = train_data.sort_values('product_rating', ascending=False).head(5)


    # Use TF-IDF Vectorizer for similarities
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(train_data['product_name'])

    # Compute cosine similarities
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index of the first matched product
    idx = matched_products.index[0]
    app.logger.debug(f"Index of matched product: {idx}")

    # Get similar products based on cosine similarity
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]

    # Get product indices
    product_indices = [i[0] for i in sim_scores]

    # Fetch recommended products
    recommended_products = train_data.iloc[product_indices]
    app.logger.debug(f"Recommended products: {recommended_products.head()}")

    # Handle 'No rating available' by replacing it with 0.0
    recommended_products['product_rating'] = recommended_products['product_rating'].apply(
        lambda x: float(x) if x != 'No rating available' else 0.0
    )

    return recommended_products[['product_name', 'brand', 'image', 'product_rating', 'retail_price','discounted_price']]

# Assuming signup_table contains user_id and relevant user details

user_item_matrix = train_data.pivot_table(
    index='uniq_id',
    columns='pid',
    values='product_rating',
    fill_value=0
)


def collaborative_recommendations(train_data, user_item_matrix, product_id, top_n=20):
    # Find the index of the product
    product_idx = train_data[train_data['product_id'] == product_id].index[0]

    # Perform matrix factorization (SVD)
    svd = TruncatedSVD(n_components=50)
    user_item_matrix_sparse = csr_matrix(user_item_matrix)
    latent_matrix = svd.fit_transform(user_item_matrix_sparse)

    # Compute similarity scores
    similarity_scores = cosine_similarity(latent_matrix[product_idx].reshape(1, -1), latent_matrix).flatten()

    # Get top N similar products
    similar_indices = np.argsort(-similarity_scores)[1:top_n + 1]
    recommended_products = train_data.iloc[similar_indices]

    return recommended_products[
        ['product_name', 'brand', 'image', 'product_rating', 'retail_price', 'discounted_price']]


def hybrid_recommendations(train_data, user_item_matrix, item_name, top_n=20):
    content_recs = content_based_recommendations(train_data, item_name, top_n)
    collab_recs = collaborative_recommendations(train_data, user_item_matrix, item_name, top_n)
    merged_recs = pd.concat([content_recs, collab_recs]).drop_duplicates()
    return merged_recs.head(top_n)

#routes
@app.route("/")
def index():
    """Main page to display trending products or recommendations"""
    username = session.get('username')
    user_id = session.get('user_id')

    # Fetch top-rated products and convert to a list of dictionaries
    trending_subset = get_top_rated_products(28).to_dict(orient='records') if get_top_rated_products(28) is not None else []

    recommendations = []
    if user_id:
        # Provide recommendations if the user is signed in
        recommendations = train_data.sort_values('product_rating', ascending=False).head(5).to_dict(orient='records')

    # Render the index page
    return render_template(
        "index.html",
        username=username,
        trending_products=trending_subset,
        recommendations=recommendations,
    )

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    """ Handle recommendations based on user input """
    if request.method == 'POST':
        prod = request.form.get('prod', '').strip()

        if not prod:
            message = "Please enter a product name."
            return render_template('main.html', message=message)

        # Log the search activity
        user_id = session.get('user_id')
        if user_id:
            insert_user_activity(user_id, prod, "search")  # Log search activity

        # Fetch recommendations based on the product name using the dataset
        recommended_products = train_data[train_data['product_name'].str.contains(prod, case=False, na=False)]

        if recommended_products.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message)
        else:
            recommended_product_info = recommended_products.to_dict(orient='records')
            return render_template('main.html', content_based_rec=recommended_product_info)

    return render_template('main.html')


@app.route('/main')
def main():
    username = session.get('username')
    user_id = session.get('user_id')
    return render_template('main.html',username=username)



@app.route("/index")
def indexredirect():
    return redirect(url_for('index'))


@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if Signup.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.', 'error')
            return redirect(url_for('signup'))
        if Signup.query.filter_by(email=email).first():
            flash('Email already registered. Please use another email.', 'error')
            return redirect(url_for('signup'))

        # Store the original password in the database
        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()
        flash('Signup successful! You can now log in.', 'success')
        return redirect(url_for('signin'))
    return render_template('signup.html')


@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']

        existing_user = Signup.query.filter_by(username=username).first()
        if existing_user:
            if existing_user.password == password:  # Consider hashing for security
                session.clear()  # Clear existing session
                session['user_id'] = existing_user.id  # Store user ID in session
                session['username'] = existing_user.username  # Store username in session
                flash('User signed in successfully!', 'success')
                return redirect(url_for('index'))  # Redirect to index after successful sign-in
            else:
                flash('Invalid password. Please try again.', 'error')
        else:
            flash('Invalid username. Please try again.', 'error')

    return render_template('signin.html')

@app.route('/logout')
def logout():
    session.clear()  # Clear the session on logout
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)