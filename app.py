import subprocess
import sys
import os

# פונקציה להתקנת תלות
def install_dependencies():
    required_packages = ['flask', 'flask-wtf', 'flask-limiter', 'requests', 'flask-mail', 'pyotp']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"התוסף {package} לא מותקן. מתקין...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# התקן תלות אוטומטית
install_dependencies()

from flask import Flask, render_template_string, request, session, redirect, url_for, flash, jsonify, abort
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
import sqlite3
import re
import datetime
import requests
import csv
import logging
import pyotp
import unittest

app = Flask(__name__)
app.secret_key = os.urandom(24)  # מפתח אבטחה אקראי
app.config['SESSION_COOKIE_SECURE'] = True  # Secure flag for cookies
app.config['SESSION_COOKIE_HTTPONLY'] = True  # HttpOnly flag for cookies
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # זמן תוקף סשן (30 דקות)

# הוספת תמיכה באימייל
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your-email-password'
mail = Mail(app)

# הגנה מפני CSRF
csrf = CSRFProtect(app)

# Rate Limiting להגנה מפני Brute Force
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# מסד נתונים לדוגמה (SQLite)
DATABASE = 'app.db'

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
                role TEXT DEFAULT 'user'
            )
        ''')
        conn.commit()

# פונקציה להצפנת סיסמאות
def hash_password(password):
    return generate_password_hash(password)

# פונקציה לבדיקת סיסמאות
def verify_password(password_hash, password):
    return check_password_hash(password_hash, password)

# פונקציה לסינון קלט
def sanitize_input(input_str):
    return re.sub(r'[<>"\']', '', input_str)  # הסרת תווים מסוכנים

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

# דף נחיתה
@app.route('/')
def landing_page():
    language = session.get('language', 'he')  # ברירת מחדל: עברית
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

# שינוי שפה
@app.route('/set_language/<language>')
def set_language(language):
    if language in LANGUAGES:
        session['language'] = language
    return redirect(request.referrer or url_for('landing_page'))

# דף התחברות
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
            cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()

            if user and verify_password(user[0], password):
                session['username'] = username
                flash('התחברת בהצלחה!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('שם משתמש או סיסמה לא נכונים', 'error')
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

# דף הרשמה
@app.route('/register', methods=['GET', 'POST'])
def register():
    language = session.get('language', 'he')
    lang = LANGUAGES[language]
    
    if request.method == 'POST':
        username = sanitize_input(request.form['username'])
        email = sanitize_input(request.form['email'])
        password = request.form['password']
        password_hash = hash_password(password)

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)', (username, email, password_hash))
                conn.commit()
                flash('נרשמת בהצלחה! אנא אמת את האימייל שלך.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('שם משתמש או אימייל כבר קיימים', 'error')
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
                        <label for="email" class="block text-right">אימייל:</label>
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

# דף יציאה
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('יצאת בהצלחה!', 'success')
    return redirect(url_for('landing_page'))

# פונקציה לאחזור פגיעויות מ-NVD
def fetch_vulnerabilities():
    start_date = (datetime.date.today() - datetime.timedelta(days=30)).isoformat() + "T00:00:00.000"
    end_date = datetime.date.today().isoformat() + "T23:59:59.999"
    params = {
        "pubStartDate": start_date,
        "pubEndDate": end_date,
        "resultsPerPage": 50
    }

    try:
        response = requests.get("https://services.nvd.nist.gov/rest/json/cves/2.0", params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('vulnerabilities', [])
        else:
            print(f"API Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
    
    return []

# פונקציה ליצירת פלייבוק הגנה
def generate_playbook(vuln):
    playbook = {
        "steps": [],
        "products": [],
        "description": vuln.get('cve', {}).get('descriptions', [{}])[0].get('value', 'אין תיאור'),
        "severity": vuln.get('cve', {}).get('metrics', {}).get('cvssMetricV31', [{}])[0].get('cvssData', {}).get('baseSeverity', 'N/A')
    }
    
    # מציאת מוצרים מושפעים
    if vuln.get('cve', {}).get('configurations'):
        for config in vuln['cve']['configurations']:
            if config.get('nodes'):
                for node in config['nodes']:
                    if node.get('cpeMatch'):
                        for cpe in node['cpeMatch']:
                            if cpe.get('criteria'):
                                cpe_parts = cpe['criteria'].split(':')
                                if len(cpe_parts) > 4:
                                    vendor = cpe_parts[3]
                                    product = cpe_parts[4]
                                    version = cpe_parts[5] if len(cpe_parts) > 5 else ""
                                    if vendor and product:
                                        product_info = f"{vendor} {product}"
                                        if version and version != "*":
                                            product_info += f" {version}"
                                        if product_info not in playbook["products"]:
                                            playbook["products"].append(product_info)
    
    # צעדים ספציפיים לפי סוג החולשה
    description = playbook["description"].lower()
    if re.search(r'buffer\s+overflow|stack\s+overflow', description):
        playbook["steps"].extend([
            "עדכן את התוכנה לגרסה האחרונה שמתקנת את החולשה.",
            "הפעל מנגנוני הגנה כמו DEP ו-ASLR.",
            "בצע סקירת קוד למציאת חולשות דומות.",
            "השתמש בכלים אוטומטיים לזיהוי חולשות בזיכרון."
        ])
    elif re.search(r'sql\s+injection', description):
        playbook["steps"].extend([
            "השתמש בשאילתות מוכנות מראש (Prepared Statements) או ORM.",
            "הוסף שכבת סינון קלט (Input Validation) לכל הקלטים מהמשתמש.",
            "הגבל הרשאות של המשתמש בבסיס הנתונים.",
            "בצע בדיקות חדירה כדי לוודא שהמערכת מוגנת."
        ])
    elif re.search(r'xss|cross.?site\s+scripting', description):
        playbook["steps"].extend([
            "טפל בכל קלט משתמש לפני הצגתו בדף (HTML Escaping).",
            "הפעל מדיניות CSP (Content Security Policy) קפדנית.",
            "בדוק את כל נקודות הקלט בתוכנה."
        ])
    elif re.search(r'remote\s+code\s+execution|rce', description):
        playbook["steps"].extend([
            "עדכן מיידית את התוכנה לגרסה האחרונה.",
            "הפעל מנגנוני Sandboxing והרשאות מינימליות.",
            "נטר לוגים לאיתור פעילות חשודה."
        ])
    else:
        # צעדים כלליים אם לא נמצאו צעדים ספציפיים
        playbook["steps"] = [
            "עדכן את התוכנה לגרסה האחרונה.",
            "הגבל הרשאות גישה למינימום הנדרש.",
            "בדוק את התצורה של המערכת להסרת הגדרות לא בטוחות.",
            "נטר לוגים של המערכת לזיהוי פעילות חשודה."
        ]
    
    # הוסף צעד אחרון תמיד
    playbook["steps"].append("בצע בדיקות חדירות לאחר יישום התיקונים.")
    
    return playbook

# דשבורד פגיעויות
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    language = session.get('language', 'he')
    lang = LANGUAGES[language]
    
    vulnerabilities = fetch_vulnerabilities()
    
    # ספירת רמות החומרה
    critical_count = 0
    high_count = 0
    
    for vuln in vulnerabilities:
        if vuln.get('cve', {}).get('metrics', {}).get('cvssMetricV31'):
            severity = vuln['cve']['metrics']['cvssMetricV31'][0]['cvssData']['baseSeverity']
            if severity == 'CRITICAL':
                critical_count += 1
            elif severity == 'HIGH':
                high_count += 1
    
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="{{ language }}" dir="{{ 'rtl' if language == 'he' else 'ltr' }}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ lang['dashboard_title'] }}</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css">
        </head>
        <body class="bg-gray-100 text-gray-900">
            <div class="container mx-auto p-6">
                <h1 class="text-3xl font-bold mb-6 text-right">{{ lang['dashboard_title'] }}</h1>
                
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    <div class="bg-white p-4 rounded-lg shadow">
                        <h2 class="text-lg font-semibold mb-2 text-right">{{ lang['total_vulnerabilities'] }}</h2>
                        <p class="text-3xl font-bold text-center">{{ vulnerabilities|length }}</p>
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
                        <p class="text-sm text-center">30 הימים האחרונים</p>
                    </div>
                </div>
                
                <div class="bg-white p-4 rounded-lg shadow mb-6">
                    <input type="text" id="search" placeholder="{{ lang['search_placeholder'] }}" 
                           class="w-full p-2 border border-gray-300 rounded-lg text-right">
                </div>
                
                <div class="overflow-x-auto bg-white rounded-lg shadow">
                    <table class="min-w-full">
                        <thead>
                            <tr class="bg-gray-50">
                                <th class="py-2 px-4 border-b text-right">{{ lang['vulnerability_id'] }}</th>
                                <th class="py-2 px-4 border-b text-right">{{ lang['description'] }}</th>
                                <th class="py-2 px-4 border-b text-right">{{ lang['severity'] }}</th>
                                <th class="py-2 px-4 border-b text-right">{{ lang['actions'] }}</th>
                            </tr>
                        </thead>
                        <tbody id="vulnTable">
                            {% if vulnerabilities %}
                                {% for vuln in vulnerabilities %}
                                {% set severity = vuln.cve.metrics.cvssMetricV31[0].cvssData.baseSeverity if vuln.cve.metrics and vuln.cve.metrics.cvssMetricV31 else 'N/A' %}
                                <tr data-vuln-id="{{ vuln.cve.id }}">
                                    <td class="py-2 px-4 border-b">{{ vuln.cve.id }}</td>
                                    <td class="py-2 px-4 border-b text-right">
                                        {{ vuln.cve.descriptions[0].value[:150] + '...' if vuln.cve.descriptions and vuln.cve.descriptions[0].value|length > 150 else vuln.cve.descriptions[0].value if vuln.cve.descriptions else 'אין תיאור' }}
                                    </td>
                                    <td class="py-2 px-4 border-b text-center">
                                        <span class="px-2 py-1 rounded {% if severity == 'CRITICAL' %}bg-red-100 text-red-800{% elif severity == 'HIGH' %}bg-amber-100 text-amber-800{% elif severity == 'MEDIUM' %}bg-blue-100 text-blue-800{% elif severity == 'LOW' %}bg-green-100 text-green-800{% endif %}">
                                        {% if severity == 'CRITICAL' %}חמורה{% elif severity == 'HIGH' %}גבוהה{% elif severity == 'MEDIUM' %}בינונית{% elif severity == 'LOW' %}נמוכה{% else %}לא ידוע{% endif %}
                                        </span>
                                    </td>
                                    <td class="py-2 px-4 border-b text-center">
                                        <button class="bg-green-100 text-green-800 px-2 py-1 rounded hover:bg-green-200 mr-2"
                                                onclick="showPlaybook('{{ vuln.cve.id }}')">{{ lang['playbook'] }}</button>
                                        <a href="https://nvd.nist.gov/vuln/detail/{{ vuln.cve.id }}" 
                                           class="text-blue-500 hover:underline" target="_blank">{{ lang['more_details'] }}</a>
                                    </td>
                                </tr>
                                {% endfor %}
                            {% else %}
                                <tr>
                                    <td colspan="4" class="py-2 px-4 border-b text-center">{{ lang['no_vulnerabilities'] }}</td>
                                </tr>
                            {% endif %}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Modal לפלייבוק -->
            <div id="playbookModal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closePlaybook()">&times;</span>
                    <h2 id="playbookTitle" class="text-2xl font-bold mb-4 text-right"></h2>
                    
                    <div class="mb-4">
                        <h3 class="text-lg font-semibold mb-2 text-right">{{ lang['description'] }}:</h3>
                        <p id="vulnDescription" class="text-right"></p>
                    </div>
                    
                    <div class="mb-4">
                        <h3 class="text-lg font-semibold mb-2 text-right">{{ lang['affected_products'] }}:</h3>
                        <ul id="productsList" class="list-disc mr-8 text-right"></ul>
                    </div>
                    
                    <div>
                        <h3 class="text-lg font-semibold mb-2 text-right">{{ lang['recommended_steps'] }}:</h3>
                        <ol id="playbookSteps" class="list-decimal mr-8 space-y-2 text-right"></ol>
                    </div>
                </div>
            </div>
            
            <script>
                // נתוני הפלייבוק
                const playbooks = {
                    {% for vuln in vulnerabilities %}
                    "{{ vuln.cve.id }}": {{ generate_playbook(vuln)|tojson }}{% if not loop.last %},{% endif %}
                    {% endfor %}
                };
            
                // חיפוש לפי טקסט
                const searchInput = document.getElementById('search');
                searchInput.addEventListener('input', function() {
                    const rows = document.querySelectorAll('#vulnTable tr');
                    const query = searchInput.value.toLowerCase();
                    
                    rows.forEach(row => {
                        const text = row.innerText.toLowerCase();
                        row.style.display = text.includes(query) ? '' : 'none';
                    });
                });
                
                // מודל הפלייבוק
                const modal = document.getElementById("playbookModal");
                
                function showPlaybook(id) {
                    const playbook = playbooks[id];
                    const title = document.getElementById("playbookTitle");
                    const description = document.getElementById("vulnDescription");
                    const steps = document.getElementById("playbookSteps");
                    const products = document.getElementById("productsList");
                    
                    title.textContent = `{{ lang['playbook'] }} - ${id}`;
                    description.textContent = playbook.description || "{{ lang['no_description'] }}";
                    
                    // הצגת המוצרים המושפעים
                    products.innerHTML = '';
                    if (playbook.products && playbook.products.length > 0) {
                        playbook.products.forEach(product => {
                            const li = document.createElement("li");
                            li.textContent = product;
                            products.appendChild(li);
                        });
                    } else {
                        const li = document.createElement("li");
                        li.textContent = "{{ lang['no_products'] }}";
                        products.appendChild(li);
                    }
                    
                    // הצגת צעדי הגנה
                    steps.innerHTML = '';
                    playbook.steps.forEach(step => {
                        const li = document.createElement("li");
                        li.textContent = step;
                        steps.appendChild(li);
                    });
                    
                    modal.style.display = "block";
                }
                
                function closePlaybook() {
                    modal.style.display = "none";
                }
                
                // סגירת המודל כאשר לוחצים מחוץ לו
                window.onclick = function(event) {
                    if (event.target == modal) {
                        closePlaybook();
                    }
                }
            </script>
        </body>
        </html>
    ''', vulnerabilities=vulnerabilities, critical_count=critical_count, high_count=high_count, generate_playbook=generate_playbook, lang=lang, language=language)

# API לאחזור פגיעויות
@app.route('/api/vulnerabilities', methods=['GET'])
def get_vulnerabilities():
    vulnerabilities = fetch_vulnerabilities()
    return jsonify(vulnerabilities)

# API להוספת פגיעות
@app.route('/api/vulnerabilities', methods=['POST'])
def add_vulnerability():
    data = request.json
    # כאן נוסיף לוגיקה לשמירת הפגיעות במסד הנתונים
    return jsonify({"status": "success", "id": new_vulnerability_id})

# API לעדכון פגיעות
@app.route('/api/vulnerabilities/<vuln_id>', methods=['PUT'])
def update_vulnerability(vuln_id):
    data = request.json
    # כאן נוסיף לוגיקה לעדכון הפגיעות במסד הנתונים
    return jsonify({"status": "success"})

# בדיקות יחידה
class TestApp(unittest.TestCase):
    def test_password_hashing(self):
        password = "securepassword"
        hashed = hash_password(password)
        self.assertTrue(verify_password(hashed, password))
        self.assertFalse(verify_password(hashed, "wrongpassword"))

if __name__ == '__main__':
    init_db()
    app.run(debug=False)  # ודא ש-debug מושבת בסביבת Production
