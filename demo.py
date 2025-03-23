import subprocess
import sys
import os
from flask import Flask, render_template_string, request, session, redirect, url_for, flash, send_file, Response
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
import sqlite3
import re
import datetime
import requests
import logging
import pyotp
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
from io import BytesIO
from math import ceil

# Install missing dependencies
def install_dependencies():
    """
    Install required Python packages if they are missing.
    """
    required_packages = [
        "flask",
        "flask-wtf",
        "flask-limiter",
        "werkzeug",
        "requests",
        "pyotp",
        "flask-mail",
        "apscheduler",
        "pandas",
        "openpyxl"
    ]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Function to generate a configuration file if it doesn't exist
def generate_config_file():
    """
    Generate a configuration file with default values if it doesn't exist.
    """
    config_content = """# Configuration file
NIST_API_KEY = "2849cc87-6d85-4110-b972-03bdb9218264"  # Your NIST API key
MAIL_USERNAME = "your_email@gmail.com"  # Replace with your email
MAIL_PASSWORD = "your_email_password"  # Replace with your email password
"""

    if not os.path.exists("config.py"):
        print("Generating config.py...")
        with open("config.py", "w") as f:
            f.write(config_content)

# Install dependencies and generate config file
install_dependencies()
generate_config_file()

# Now import the rest of the libraries
from config import NIST_API_KEY, MAIL_USERNAME, MAIL_PASSWORD

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Random security key
app.config['SESSION_COOKIE_SECURE'] = True  # Secure flag for cookies
app.config['SESSION_COOKIE_HTTPONLY'] = True  # HttpOnly flag for cookies
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # Session lifetime (30 minutes)

# Email support
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = MAIL_USERNAME  # Use environment variables for sensitive data
app.config['MAIL_PASSWORD'] = MAIL_PASSWORD
mail = Mail(app)

# CSRF protection
csrf = CSRFProtect(app)

# Rate Limiting for Brute Force protection
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Database (SQLite)
DATABASE = 'app.db'

# מילון לשפות
LANGUAGES = {
    'en': {
        'welcome': 'Welcome to the Cybersecurity Vulnerability Management System',
        'login': 'Login',
        'register': 'Register',
        'username': 'Username',
        'password': 'Password',
        'submit': 'Submit',
        'no_account': 'Don\'t have an account?',
        'have_account': 'Already have an account?',
        'sign_up': 'Sign Up',
        'sign_in': 'Sign In',
        'logout': 'Logout',
        'dashboard_title': 'Cybersecurity Vulnerabilities Dashboard',
        'total_vulnerabilities': 'Total Vulnerabilities',
        'critical': 'Critical',
        'high': 'High',
        'time_range': 'Last 30 Days',
        'search_placeholder': 'Search vulnerabilities...',
        'vulnerability_id': 'ID',
        'description': 'Description',
        'severity': 'Severity',
        'actions': 'Actions',
        'playbook': 'Defense Playbook',
        'more_details': 'More Details',
        'affected_products': 'Affected Products',
        'recommended_steps': 'Recommended Steps',
        'no_vulnerabilities': 'No vulnerabilities found',
    },
    'he': {
        'welcome': 'ברוכים הבאים למערכת ניהול פגיעויות סייבר',
        'login': 'התחברות',
        'register': 'הרשמה',
        'username': 'שם משתמש',
        'password': 'סיסמה',
        'submit': 'שלח',
        'no_account': 'אין לך חשבון?',
        'have_account': 'כבר יש לך חשבון?',
        'sign_up': 'הירשם',
        'sign_in': 'התחבר',
        'logout': 'יציאה',
        'dashboard_title': 'דשבורד פגיעויות סייבר',
        'total_vulnerabilities': 'סה"כ פגיעויות',
        'critical': 'חמורות',
        'high': 'גבוהות',
        'time_range': '30 הימים האחרונים',
        'search_placeholder': 'חפש פגיעויות...',
        'vulnerability_id': 'מזהה',
        'description': 'תיאור',
        'severity': 'חומרה',
        'actions': 'פעולות',
        'playbook': 'פלייבוק הגנה',
        'more_details': 'פרטים נוספים',
        'affected_products': 'מוצרים מושפעים',
        'recommended_steps': 'צעדי הגנה מומלצים',
        'no_vulnerabilities': 'לא נמצאו פגיעויות',
    }
}

# Initialize database
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email_verified BOOLEAN DEFAULT FALSE,
                role TEXT DEFAULT 'user',
                mfa_secret TEXT
            )
        ''')
        conn.commit()

# Password hashing
def hash_password(password):
    return generate_password_hash(password)

# Password verification
def verify_password(password_hash, password):
    return check_password_hash(password_hash, password)

# Input sanitization
def sanitize_input(input_str):
    return re.sub(r'[<>"\']', '', input_str)  # Remove dangerous characters

# Generate MFA Secret
def generate_mfa_secret():
    return pyotp.random_base32()

# Verify MFA Code
def verify_mfa_code(secret, code):
    totp = pyotp.TOTP(secret)
    return totp.verify(code)

# Fetch vulnerabilities from NVD
def fetch_vulnerabilities():
    """
    Fetch vulnerabilities from the National Vulnerability Database (NVD) API using the provided API key.
    """
    headers = {
        "apiKey": NIST_API_KEY
    }

    start_date = (datetime.date.today() - datetime.timedelta(days=30)).isoformat() + "T00:00:00.000"
    end_date = datetime.date.today().isoformat() + "T23:59:59.999"
    params = {
        "pubStartDate": start_date,
        "pubEndDate": end_date,
        "resultsPerPage": 50
    }

    try:
        response = requests.get(
            "https://services.nvd.nist.gov/rest/json/cves/2.0",
            headers=headers,
            params=params
        )
        if response.status_code == 200:
            data = response.json()
            vulnerabilities = data.get('vulnerabilities', [])
            
            # Add OWASP Top 10 and MITRE ATT&CK categorization to each vulnerability
            for vuln in vulnerabilities:
                description = vuln.get('cve', {}).get('descriptions', [{}])[0].get('value', 'No description')
                vuln['owasp_category'] = map_to_owasp_top_10(description)
                vuln['mitre_attack'] = map_to_mitre_attack(description)  # Add MITRE ATT&CK mapping

                # Extract patch links from references
                references = vuln.get('cve', {}).get('references', [])
                patch_links = [ref['url'] for ref in references if 'Patch' in ref.get('tags', [])]
                vuln['patch_links'] = patch_links  # Save patch links

            return vulnerabilities
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Exception: {e}")
        return []

# Schedule the fetch_vulnerabilities function to run every 24 hours
scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(
    func=fetch_vulnerabilities,
    trigger="interval",
    hours=24,
    id="fetch_vulnerabilities_job"
)

# Map vulnerability to OWASP Top 10 category
def map_to_owasp_top_10(vulnerability_description):
    """
    Maps a vulnerability description to an OWASP Top 10 category based on keywords.
    """
    description = vulnerability_description.lower()

    if any(keyword in description for keyword in ["access control", "unauthorized access"]):
        return "A01:2021 - Broken Access Control"
    elif any(keyword in description for keyword in ["cryptographic", "encryption", "sensitive data"]):
        return "A02:2021 - Cryptographic Failures"
    elif any(keyword in description for keyword in ["injection", "sql injection", "nosql injection", "os injection"]):
        return "A03:2021 - Injection"
    elif any(keyword in description for keyword in ["insecure design", "design flaw"]):
        return "A04:2021 - Insecure Design"
    elif any(keyword in description for keyword in ["misconfiguration", "configuration error"]):
        return "A05:2021 - Security Misconfiguration"
    elif any(keyword in description for keyword in ["vulnerable component", "outdated library", "unsupported version"]):
        return "A06:2021 - Vulnerable and Outdated Components"
    elif any(keyword in description for keyword in ["authentication failure", "session management", "weak password"]):
        return "A07:2021 - Identification and Authentication Failures"
    elif any(keyword in description for keyword in ["integrity failure", "code signing", "tampering"]):
        return "A08:2021 - Software and Data Integrity Failures"
    elif any(keyword in description for keyword in ["logging failure", "monitoring failure", "incident response"]):
        return "A09:2021 - Security Logging and Monitoring Failures"
    elif any(keyword in description for keyword in ["ssrf", "server-side request forgery"]):
        return "A10:2021 - Server-Side Request Forgery (SSRF)"
    else:
        return "Uncategorized"

# Map vulnerability to MITRE ATT&CK techniques
def map_to_mitre_attack(vulnerability_description):
    """
    Maps a vulnerability description to MITRE ATT&CK techniques based on keywords.
    """
    description = vulnerability_description.lower()

    # Example mapping based on keywords
    if any(keyword in description for keyword in ["phishing", "credential harvesting"]):
        return "T1566: Phishing"
    elif any(keyword in description for keyword in ["brute force", "password spraying"]):
        return "T1110: Brute Force"
    elif any(keyword in description for keyword in ["privilege escalation", "sudo"]):
        return "T1068: Exploitation for Privilege Escalation"
    elif any(keyword in description for keyword in ["remote code execution", "rce"]):
        return "T1203: Exploitation for Client Execution"
    elif any(keyword in description for keyword in ["sql injection", "nosql injection"]):
        return "T1190: Exploit Public-Facing Application"
    elif any(keyword in description for keyword in ["lateral movement", "pass the hash"]):
        return "T1072: Lateral Movement"
    elif any(keyword in description for keyword in ["data exfiltration", "data theft"]):
        return "T1041: Exfiltration Over C2 Channel"
    else:
        return "Uncategorized"

# Landing page
@app.route('/')
def landing_page():
    language = session.get('language', 'he')  # Default: Hebrew
    lang = LANGUAGES[language]
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="{{ language }}" dir="{{ 'rtl' if language == 'he' else 'ltr' }}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ lang['welcome'] }}</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css">
        </head>
        <body class="bg-gray-100 text-gray-900">
            <div class="container mx-auto p-6 text-center">
                <h1 class="text-3xl font-bold mb-6">{{ lang['welcome'] }}</h1>
                <div class="space-x-4">
                    <a href="{{ url_for('login') }}" class="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600">{{ lang['login'] }}</a>
                    <a href="{{ url_for('register') }}" class="bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600">{{ lang['register'] }}</a>
                </div>
                <div class="mt-4">
                    <a href="{{ url_for('set_language', language='he') }}" class="text-blue-500 hover:underline">עברית</a> |
                    <a href="{{ url_for('set_language', language='en') }}" class="text-blue-500 hover:underline">English</a>
                </div>
            </div>
        </body>
        </html>
    ''', lang=lang, language=language)

# Set language
@app.route('/set_language/<language>')
def set_language(language):
    if language in LANGUAGES:
        session['language'] = language
    return redirect(request.referrer or url_for('landing_page'))

# Login page
@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    language = session.get('language', 'he')
    lang = LANGUAGES[language]
    
    if request.method == 'POST':
        username = sanitize_input(request.form['username'])
        password = request.form['password']

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT password_hash, mfa_secret FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()

            if user and verify_password(user[0], password):
                session['username'] = username
                if user[1]:  # If MFA is enabled
                    return redirect(url_for('mfa_verify'))
                flash('Logged in successfully!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Incorrect username or password', 'error')
                return redirect(url_for('login'))

    return render_template_string('''
        <!DOCTYPE html>
        <html lang="{{ language }}" dir="{{ 'rtl' if language == 'he' else 'ltr' }}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ lang['login'] }}</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css">
        </head>
        <body class="bg-gray-100 text-gray-900">
            <div class="container mx-auto p-6">
                <h1 class="text-3xl font-bold mb-6 text-right">{{ lang['login'] }}</h1>
                <form method="post" class="bg-white p-6 rounded-lg shadow-md">
                    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                    <div class="mb-4">
                        <label for="username" class="block text-right">{{ lang['username'] }}:</label>
                        <input type="text" id="username" name="username" class="w-full p-2 border border-gray-300 rounded-lg text-right" required>
                    </div>
                    <div class="mb-4">
                        <label for="password" class="block text-right">{{ lang['password'] }}:</label>
                        <input type="password" id="password" name="password" class="w-full p-2 border border-gray-300 rounded-lg text-right" required>
                    </div>
                    <button type="submit" class="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 w-full">{{ lang['submit'] }}</button>
                </form>
                <p class="mt-4 text-right">{{ lang['no_account'] }} <a href="{{ url_for('register') }}" class="text-blue-500 hover:underline">{{ lang['sign_up'] }}</a></p>
                <div class="mt-4">
                    <a href="{{ url_for('set_language', language='he') }}" class="text-blue-500 hover:underline">עברית</a> |
                    <a href="{{ url_for('set_language', language='en') }}" class="text-blue-500 hover:underline">English</a>
                </div>
            </div>
        </body>
        </html>
    ''', lang=lang, language=language)

# MFA Verification
@app.route('/mfa_verify', methods=['GET', 'POST'])
def mfa_verify():
    if 'username' not in session:
        flash('You must log in first', 'error')
        return redirect(url_for('login'))

    language = session.get('language', 'he')
    lang = LANGUAGES[language]

    if request.method == 'POST':
        code = request.form.get('code')
        if not code:
            flash('MFA code is required', 'error')
            return redirect(url_for('mfa_verify'))

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT mfa_secret FROM users WHERE username = ?', (session['username'],))
            result = cursor.fetchone()

            if result and result[0]:  # Check if MFA secret exists
                mfa_secret = result[0]
                if verify_mfa_code(mfa_secret, code):
                    session['mfa_verified'] = True  # Set MFA verification flag
                    flash('MFA verification successful!', 'success')
                    return redirect(url_for('dashboard'))  # Redirect to dashboard
                else:
                    flash('Invalid MFA code', 'error')
                    return redirect(url_for('mfa_verify'))
            else:
                flash('MFA secret not found for this user. Please contact support.', 'error')
                return redirect(url_for('mfa_verify'))

    return render_template_string('''
        <!DOCTYPE html>
        <html lang="{{ language }}" dir="{{ 'rtl' if language == 'he' else 'ltr' }}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MFA Verification</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css">
        </head>
        <body class="bg-gray-100 text-gray-900">
            <div class="container mx-auto p-6">
                <h1 class="text-3xl font-bold mb-6 text-right">MFA Verification</h1>
                <form method="post" class="bg-white p-6 rounded-lg shadow-md">
                    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                    <div class="mb-4">
                        <label for="code" class="block text-right">Enter 6-digit MFA Code:</label>
                        <input type="text" id="code" name="code" class="w-full p-2 border border-gray-300 rounded-lg text-right" required>
                    </div>
                    <button type="submit" class="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 w-full">Verify</button>
                </form>
            </div>
        </body>
        </html>
    ''', lang=lang, language=language)

# Register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    language = session.get('language', 'he')
    lang = LANGUAGES[language]
    
    if request.method == 'POST':
        username = sanitize_input(request.form['username'])
        email = sanitize_input(request.form['email'])
        password = request.form['password']
        password_hash = hash_password(password)
        mfa_secret = generate_mfa_secret()

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('INSERT INTO users (username, email, password_hash, mfa_secret) VALUES (?, ?, ?, ?)', 
                              (username, email, password_hash, mfa_secret))
                conn.commit()

                # Save the MFA key to a local text file
                mfa_key_file_path = os.path.join(os.getcwd(), f"{username}_mfa_key.txt")
                with open(mfa_key_file_path, 'w') as f:
                    f.write(f"Username: {username}\n")
                    f.write(f"MFA Secret Key: {mfa_secret}\n")
                    f.write("Instructions:\n")
                    f.write("1. Install an authenticator app (e.g., Google Authenticator).\n")
                    f.write("2. Scan the QR code or manually enter the MFA secret.\n")
                    f.write("3. Use the 6-digit code from the app to log in.\n")

                flash(f'Registered successfully! Your MFA key has been saved to {mfa_key_file_path}. Store it securely.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username or email already exists', 'error')
                return redirect(url_for('register'))

    return render_template_string('''
        <!DOCTYPE html>
        <html lang="{{ language }}" dir="{{ 'rtl' if language == 'he' else 'ltr' }}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ lang['register'] }}</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css">
        </head>
        <body class="bg-gray-100 text-gray-900">
            <div class="container mx-auto p-6">
                <h1 class="text-3xl font-bold mb-6 text-right">{{ lang['register'] }}</h1>
                <form method="post" class="bg-white p-6 rounded-lg shadow-md">
                    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                    <div class="mb-4">
                        <label for="username" class="block text-right">{{ lang['username'] }}:</label>
                        <input type="text" id="username" name="username" class="w-full p-2 border border-gray-300 rounded-lg text-right" required>
                    </div>
                    <div class="mb-4">
                        <label for="email" class="block text-right">Email:</label>
                        <input type="email" id="email" name="email" class="w-full p-2 border border-gray-300 rounded-lg text-right" required>
                    </div>
                    <div class="mb-4">
                        <label for="password" class="block text-right">{{ lang['password'] }}:</label>
                        <input type="password" id="password" name="password" class="w-full p-2 border border-gray-300 rounded-lg text-right" required>
                    </div>
                    <button type="submit" class="bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600 w-full">{{ lang['sign_up'] }}</button>
                </form>
                <p class="mt-4 text-right">{{ lang['have_account'] }} <a href="{{ url_for('login') }}" class="text-blue-500 hover:underline">{{ lang['sign_in'] }}</a></p>
                <div class="mt-4">
                    <a href="{{ url_for('set_language', language='he') }}" class="text-blue-500 hover:underline">עברית</a> |
                    <a href="{{ url_for('set_language', language='en') }}" class="text-blue-500 hover:underline">English</a>
                </div>
            </div>
        </body>
        </html>
    ''', lang=lang, language=language)

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('mfa_verified', None)  # Clear MFA flag
    flash('Logged out successfully!', 'success')
    return redirect(url_for('landing_page'))

# Dashboard
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('You must log in first', 'error')
        return redirect(url_for('login'))

    # Ensure MFA verification is completed
    if not session.get('mfa_verified'):
        flash('MFA verification required', 'error')
        return redirect(url_for('mfa_verify'))

    # Fetch vulnerabilities
    vulnerabilities = fetch_vulnerabilities()

    # Add OWASP Top 10 and MITRE ATT&CK categorization to each vulnerability
    for vuln in vulnerabilities:
        description = vuln.get('cve', {}).get('descriptions', [{}])[0].get('value', 'No description')
        vuln['owasp_category'] = map_to_owasp_top_10(description)
        vuln['mitre_attack'] = map_to_mitre_attack(description)  # Add MITRE ATT&CK mapping

        # Extract patch links from references
        references = vuln.get('cve', {}).get('references', [])
        patch_links = [ref['url'] for ref in references if 'Patch' in ref.get('tags', [])]
        vuln['patch_links'] = patch_links  # Save patch links

    # Pagination logic
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of vulnerabilities per page
    total_vulnerabilities = len(vulnerabilities)
    total_pages = ceil(total_vulnerabilities / per_page)

    # Slice vulnerabilities for the current page
    start = (page - 1) * per_page
    end = start + per_page
    vulnerabilities_page = vulnerabilities[start:end]

    # Count severity levels
    critical_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0
    for vuln in vulnerabilities:
        if vuln.get('cve', {}).get('metrics', {}).get('cvssMetricV31'):
            severity = vuln['cve']['metrics']['cvssMetricV31'][0]['cvssData']['baseSeverity']
            if severity == 'CRITICAL':
                critical_count += 1
            elif severity == 'HIGH':
                high_count += 1
            elif severity == 'MEDIUM':
                medium_count += 1
            elif severity == 'LOW':
                low_count += 1

    # Define language for the dashboard
    language = session.get('language', 'he')  # Default: Hebrew
    lang = LANGUAGES[language]  # Get the language dictionary

    return render_template_string('''
        <!DOCTYPE html>
        <html lang="{{ language }}" dir="{{ 'rtl' if language == 'he' else 'ltr' }}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ lang['dashboard_title'] }}</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                /* Modal styling */
                .modal {
                    display: none;
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0, 0, 0, 0.8);
                    justify-content: center;
                    align-items: center;
                    z-index: 1000;
                }

                .modal-content {
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    width: 80%;
                    max-width: 600px;
                    position: relative;
                }

                .modal-close {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: none;
                    border: none;
                    font-size: 20px;
                    cursor: pointer;
                }

                .modal-close:hover {
                    color: red;
                }
            </style>
            <script>
                // Function to show the playbook in a modal
                function showPlaybook(event, vulnerabilityId, patchLinks) {
                    event.preventDefault(); // Prevent default button behavior

                    // Fetch playbook content (you can replace this with an API call or dynamic content generation)
                    const playbookContent = generatePlaybookContent(vulnerabilityId, patchLinks);

                    // Create the modal
                    const modal = document.createElement('div');
                    modal.classList.add('modal');

                    // Create the modal content
                    const modalContent = document.createElement('div');
                    modalContent.classList.add('modal-content');

                    // Add the playbook content
                    modalContent.innerHTML = `
                        <div style="text-align: right;">
                            <button onclick="closeModal()" class="modal-close">&times;</button>
                        </div>
                        <h2 style="text-align: center; margin-bottom: 20px;">Playbook for ${vulnerabilityId}</h2>
                        <div>${playbookContent}</div>
                    `;

                    // Append modal content to the modal
                    modal.appendChild(modalContent);

                    // Append the modal to the body
                    document.body.appendChild(modal);

                    // Show the modal
                    modal.style.display = 'flex';
                }

                // Function to generate playbook content (replace with your logic)
                function generatePlaybookContent(vulnerabilityId, patchLinks) {
                    // Example content (replace with dynamic content generation)
                    let patchLinksHtml = '';
                    if (patchLinks && patchLinks.length > 0) {
                        patchLinksHtml = '<p><strong>Patch Links:</strong></p><ul>';
                        patchLinks.forEach(link => {
                            patchLinksHtml += `<li><a href="${link}" target="_blank">${link}</a></li>`;
                        });
                        patchLinksHtml += '</ul>';
                    } else {
                        patchLinksHtml = '<p>No patch links available.</p>';
                    }

                    return `
                        <p><strong>Description:</strong> This is a detailed playbook for vulnerability ${vulnerabilityId}.</p>
                        <p><strong>Steps to Mitigate:</strong></p>
                        <ol>
                            <li>Identify affected systems.</li>
                            <li>Apply the latest security patches.</li>
                            <li>Monitor for unusual activity.</li>
                        </ol>
                        <p><strong>Affected Products:</strong> Product A, Product B</p>
                        ${patchLinksHtml}
                    `;
                }

                // Function to close the modal
                function closeModal() {
                    const modal = document.querySelector('.modal');
                    if (modal) {
                        modal.remove();
                    }
                }

                // Function to initialize the chart
                function initChart() {
                    const ctx = document.getElementById('vulnerabilityChart').getContext('2d');
                    const chart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: ['Critical', 'High', 'Medium', 'Low'],
                            datasets: [{
                                label: 'Vulnerability Severity',
                                data: [{{ critical_count }}, {{ high_count }}, {{ medium_count }}, {{ low_count }}],
                                backgroundColor: [
                                    'rgba(255, 99, 132, 0.2)',
                                    'rgba(255, 159, 64, 0.2)',
                                    'rgba(75, 192, 192, 0.2)',
                                    'rgba(54, 162, 235, 0.2)'
                                ],
                                borderColor: [
                                    'rgba(255, 99, 132, 1)',
                                    'rgba(255, 159, 64, 1)',
                                    'rgba(75, 192, 192, 1)',
                                    'rgba(54, 162, 235, 1)'
                                ],
                                borderWidth: 1
                            }]
                        },
                        options: {
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }

                // Initialize the chart when the page loads
                document.addEventListener('DOMContentLoaded', initChart);
            </script>
        </head>
        <body class="bg-gray-100 text-gray-900">
            <div class="container mx-auto p-6">
                <h1 class="text-3xl font-bold mb-6 text-right">{{ lang['dashboard_title'] }}</h1>
                
                <!-- Export Buttons -->
                <div class="mb-6 flex space-x-4">
                    <a href="{{ url_for('export_excel') }}" class="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600">
                        Export to Excel
                    </a>
                    <a href="{{ url_for('export_database') }}" class="bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600">
                        Export to Database
                    </a>
                </div>
                
                <!-- Vulnerability Summary -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    <div class="bg-white p-4 rounded-lg shadow">
                        <h2 class="text-lg font-semibold mb-2 text-right">{{ lang['total_vulnerabilities'] }}</h2>
                        <p class="text-3xl font-bold text-center">{{ total_vulnerabilities }}</p>
                    </div>
                    
                    <div class="bg-white p-4 rounded-lg shadow">
                        <h2 class="text-lg font-semibold mb-2 text-right">{{ lang['critical'] }}</h2>
                        <p class="text-3xl font-bold text-center text-red-600">{{ critical_count }}</p>
                    </div>
                    
                    <div class="bg-white p-4 rounded-lg shadow">
                        <h2 class="text-lg font-semibold mb-2 text-right">{{ lang['high'] }}</h2>
                        <p class="text-3xl font-bold text-center text-amber-600">{{ high_count }}</p>
                    </div>
                    
                    <div class="bg-white p-4 rounded-lg shadow">
                        <h2 class="text-lg font-semibold mb-2 text-right">{{ lang['time_range'] }}</h2>
                        <p class="text-sm text-center">Last 30 Days</p>
                    </div>
                </div>
                
                <!-- Vulnerability Chart -->
                <div class="bg-white p-4 rounded-lg shadow mb-6">
                    <canvas id="vulnerabilityChart"></canvas>
                </div>
                
                <!-- Search Bar -->
                <div class="bg-white p-4 rounded-lg shadow mb-6">
                    <input type="text" id="search" placeholder="{{ lang['search_placeholder'] }}" 
                           class="w-full p-2 border border-gray-300 rounded-lg text-right">
                </div>
                
                <!-- Vulnerabilities Table -->
                <div class="overflow-x-auto bg-white rounded-lg shadow">
                    <table class="min-w-full">
                        <thead>
                            <tr class="bg-gray-50">
                                <th class="py-2 px-4 border-b text-right">{{ lang['vulnerability_id'] }}</th>
                                <th class="py-2 px-4 border-b text-right">{{ lang['description'] }}</th>
                                <th class="py-2 px-4 border-b text-right">{{ lang['severity'] }}</th>
                                <th class="py-2 px-4 border-b text-right">OWASP Top 10</th>
                                <th class="py-2 px-4 border-b text-right">MITRE ATT&CK</th>
                                <th class="py-2 px-4 border-b text-right">Patch Links</th>
                                <th class="py-2 px-4 border-b text-right">{{ lang['actions'] }}</th>
                            </tr>
                        </thead>
                        <tbody id="vulnTable">
                            {% if vulnerabilities_page %}
                                {% for vuln in vulnerabilities_page %}
                                {% set severity = vuln.cve.metrics.cvssMetricV31[0].cvssData.baseSeverity if vuln.cve.metrics and vuln.cve.metrics.cvssMetricV31 else 'N/A' %}
                                <tr data-vuln-id="{{ vuln.cve.id }}">
                                    <td class="py-2 px-4 border-b">{{ vuln.cve.id }}</td>
                                    <td class="py-2 px-4 border-b text-right">
                                        {{ vuln.cve.descriptions[0].value[:150] + '...' if vuln.cve.descriptions and vuln.cve.descriptions[0].value|length > 150 else vuln.cve.descriptions[0].value if vuln.cve.descriptions else 'No description' }}
                                    </td>
                                    <td class="py-2 px-4 border-b text-center">
                                        <span class="px-2 py-1 rounded {% if severity == 'CRITICAL' %}bg-red-100 text-red-800{% elif severity == 'HIGH' %}bg-amber-100 text-amber-800{% elif severity == 'MEDIUM' %}bg-blue-100 text-blue-800{% elif severity == 'LOW' %}bg-green-100 text-green-800{% endif %}">
                                        {% if severity == 'CRITICAL' %}Critical{% elif severity == 'HIGH' %}High{% elif severity == 'MEDIUM' %}Medium{% elif severity == 'LOW' %}Low{% else %}Unknown{% endif %}
                                        </span>
                                    </td>
                                    <td class="py-2 px-4 border-b text-center">
                                        {{ vuln.owasp_category }}
                                    </td>
                                    <td class="py-2 px-4 border-b text-center">
                                        {{ vuln.mitre_attack }}
                                    </td>
                                    <td class="py-2 px-4 border-b text-center">
                                        {% if vuln.patch_links %}
                                            {% for link in vuln.patch_links %}
                                                <a href="{{ link }}" class="text-blue-500 hover:underline" target="_blank">Download Patch</a><br>
                                            {% endfor %}
                                        {% else %}
                                            No patch available
                                        {% endif %}
                                    </td>
                                    <td class="py-2 px-4 border-b text-center">
                                        <button class="bg-green-100 text-green-800 px-2 py-1 rounded hover:bg-green-200 mr-2"
                                                onclick="showPlaybook(event, '{{ vuln.cve.id }}', {{ vuln.patch_links|tojson }})">{{ lang['playbook'] }}</button>
                                        <a href="https://nvd.nist.gov/vuln/detail/{{ vuln.cve.id }}" 
                                           class="text-blue-500 hover:underline" target="_blank">{{ lang['more_details'] }}</a>
                                    </td>
                                </tr>
                                {% endfor %}
                            {% else %}
                                <tr>
                                    <td colspan="7" class="py-2 px-4 border-b text-center">{{ lang['no_vulnerabilities'] }}</td>
                                </tr>
                            {% endif %}
                        </tbody>
                    </table>
                </div>
                
                <!-- Pagination -->
                <div class="mt-6 flex justify-center">
                    <nav class="inline-flex rounded-md shadow">
                        {% if page > 1 %}
                            <a href="{{ url_for('dashboard', page=page-1) }}" class="px-4 py-2 bg-white border border-gray-300 rounded-l-md hover:bg-gray-50">
                                Previous
                            </a>
                        {% endif %}
                        {% for p in range(1, total_pages + 1) %}
                            <a href="{{ url_for('dashboard', page=p) }}" class="px-4 py-2 bg-white border-t border-b border-gray-300 {% if p == page %}bg-blue-50 text-blue-600{% else %}hover:bg-gray-50{% endif %}">
                                {{ p }}
                            </a>
                        {% endfor %}
                        {% if page < total_pages %}
                            <a href="{{ url_for('dashboard', page=page+1) }}" class="px-4 py-2 bg-white border border-gray-300 rounded-r-md hover:bg-gray-50">
                                Next
                            </a>
                        {% endif %}
                    </nav>
                </div>
            </div>
        </body>
        </html>
    ''', vulnerabilities_page=vulnerabilities_page, total_vulnerabilities=total_vulnerabilities, critical_count=critical_count, high_count=high_count, medium_count=medium_count, low_count=low_count, lang=lang, language=language, page=page, total_pages=total_pages)

# Export to Excel
@app.route('/export/excel')
def export_excel():
    if 'username' not in session:
        flash('You must log in first', 'error')
        return redirect(url_for('login'))

    vulnerabilities = fetch_vulnerabilities()

    # Convert vulnerabilities to a DataFrame
    data = []
    for vuln in vulnerabilities:
        severity = vuln.get('cve', {}).get('metrics', {}).get('cvssMetricV31', [{}])[0].get('cvssData', {}).get('baseSeverity', 'N/A')
        data.append({
            'ID': vuln.get('cve', {}).get('id', 'N/A'),
            'Description': vuln.get('cve', {}).get('descriptions', [{}])[0].get('value', 'No description'),
            'Severity': severity,
            'OWASP Category': vuln.get('owasp_category', 'Uncategorized'),
            'MITRE ATT&CK': vuln.get('mitre_attack', 'Uncategorized'),
            'Patch Links': ', '.join(vuln.get('patch_links', [])) if vuln.get('patch_links') else 'No patch available'
        })

    df = pd.DataFrame(data)

    # Create an in-memory Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Vulnerabilities')

    output.seek(0)

    # Return the Excel file as a downloadable response
    return Response(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment;filename=vulnerabilities.xlsx"}
    )

# Export to Database
@app.route('/export/database')
def export_database():
    if 'username' not in session:
        flash('You must log in first', 'error')
        return redirect(url_for('login'))

    vulnerabilities = fetch_vulnerabilities()

    # Create a new SQLite database
    export_db_path = 'exported_vulnerabilities.db'
    with sqlite3.connect(export_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vulnerabilities (
                id TEXT PRIMARY KEY,
                description TEXT,
                severity TEXT,
                owasp_category TEXT,
                mitre_attack TEXT,
                patch_links TEXT
            )
        ''')

        # Insert vulnerabilities into the new database
        for vuln in vulnerabilities:
            severity = vuln.get('cve', {}).get('metrics', {}).get('cvssMetricV31', [{}])[0].get('cvssData', {}).get('baseSeverity', 'N/A')
            cursor.execute('''
                INSERT INTO vulnerabilities (id, description, severity, owasp_category, mitre_attack, patch_links)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                vuln.get('cve', {}).get('id', 'N/A'),
                vuln.get('cve', {}).get('descriptions', [{}])[0].get('value', 'No description'),
                severity,
                vuln.get('owasp_category', 'Uncategorized'),
                vuln.get('mitre_attack', 'Uncategorized'),
                ', '.join(vuln.get('patch_links', [])) if vuln.get('patch_links') else 'No patch available'
            ))

    # Return the database file as a downloadable response
    return send_file(
        export_db_path,
        as_attachment=True,
        download_name='exported_vulnerabilities.db',
        mimetype='application/x-sqlite3'
    )

# Initialize database
init_db()

if __name__ == '__main__':
    app.run(debug=False)