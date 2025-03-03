from flask import Flask, render_template_string
import requests
import datetime
import re

app = Flask(__name__)

API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

def fetch_vulnerabilities():
    start_date = (datetime.date.today() - datetime.timedelta(days=30)).isoformat() + "T00:00:00.000"
    end_date = datetime.date.today().isoformat() + "T23:59:59.999"
    params = {
        "pubStartDate": start_date,
        "pubEndDate": end_date,
        "resultsPerPage": 50
    }

    print(f"Fetching vulnerabilities from {start_date} to {end_date}")
    
    try:
        response = requests.get(API_URL, params=params)
        print(f"API Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data.get('vulnerabilities', []))} vulnerabilities")
            return data.get('vulnerabilities', [])
        else:
            print(f"API Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
    
    return []

def generate_playbook(vuln):
    """יצירת פלייבוק מותאם אישית לחולשה"""
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

# מחרוזת HTML בעברית
hebrew_template = '''<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>דשבורד פגיעויות סייבר</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .severity-critical { background-color: #FEE2E2; color: #991B1B; }
        .severity-high { background-color: #FEF3C7; color: #92400E; }
        .severity-medium { background-color: #E0F2FE; color: #075985; }
        .severity-low { background-color: #ECFCCB; color: #3F6212; }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.4);
        }
        
        .modal-content {
            background-color: #fefefe;
            margin: 10% auto;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            width: 80%;
            max-width: 800px;
        }
        
        .close {
            color: #aaa;
            float: left;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: black;
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-900">
    <div class="container mx-auto p-6">
        <h1 class="text-3xl font-bold mb-6 text-right">דשבורד פגיעויות סייבר</h1>
        
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div class="bg-white p-4 rounded-lg shadow">
                <h2 class="text-lg font-semibold mb-2 text-right">סה"כ פגיעויות</h2>
                <p class="text-3xl font-bold text-center">{{ vulnerabilities|length }}</p>
            </div>
            
            <div class="bg-white p-4 rounded-lg shadow">
                <h2 class="text-lg font-semibold mb-2 text-right">חמורות</h2>
                <p class="text-3xl font-bold text-center text-red-600">{{ critical_count }}</p>
            </div>
            
            <div class="bg-white p-4 rounded-lg shadow">
                <h2 class="text-lg font-semibold mb-2 text-right">גבוהות</h2>
                <p class="text-3xl font-bold text-center text-amber-600">{{ high_count }}</p>
            </div>
            
            <div class="bg-white p-4 rounded-lg shadow">
                <h2 class="text-lg font-semibold mb-2 text-right">טווח זמן</h2>
                <p class="text-sm text-center">30 הימים האחרונים</p>
            </div>
        </div>
        
        <div class="bg-white p-4 rounded-lg shadow mb-6">
            <input type="text" id="search" placeholder="חפש פגיעויות..." 
                   class="w-full p-2 border border-gray-300 rounded-lg text-right">
        </div>
        
        <div class="overflow-x-auto bg-white rounded-lg shadow">
            <table class="min-w-full">
                <thead>
                    <tr class="bg-gray-50">
                        <th class="py-2 px-4 border-b text-right">מזהה</th>
                        <th class="py-2 px-4 border-b text-right">תיאור</th>
                        <th class="py-2 px-4 border-b text-right">חומרה</th>
                        <th class="py-2 px-4 border-b text-right">פעולות</th>
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
                                <span class="px-2 py-1 rounded {% if severity == 'CRITICAL' %}severity-critical{% elif severity == 'HIGH' %}severity-high{% elif severity == 'MEDIUM' %}severity-medium{% elif severity == 'LOW' %}severity-low{% endif %}">
                                {% if severity == 'CRITICAL' %}חמורה{% elif severity == 'HIGH' %}גבוהה{% elif severity == 'MEDIUM' %}בינונית{% elif severity == 'LOW' %}נמוכה{% else %}לא ידוע{% endif %}
                                </span>
                            </td>
                            <td class="py-2 px-4 border-b text-center">
                                <button class="playbook-btn bg-green-100 text-green-800 px-2 py-1 rounded hover:bg-green-200 mr-2"
                                        onclick="showPlaybook('{{ vuln.cve.id }}')">פלייבוק הגנה</button>
                                <a href="https://nvd.nist.gov/vuln/detail/{{ vuln.cve.id }}" 
                                   class="text-blue-500 hover:underline" target="_blank">פרטים נוספים</a>
                            </td>
                        </tr>
                        {% endfor %}
                    {% else %}
                        <tr>
                            <td colspan="4" class="py-2 px-4 border-b text-center">לא נמצאו פגיעויות</td>
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
                <h3 class="text-lg font-semibold mb-2 text-right">תיאור החולשה:</h3>
                <p id="vulnDescription" class="text-right"></p>
            </div>
            
            <div class="mb-4">
                <h3 class="text-lg font-semibold mb-2 text-right">מוצרים מושפעים:</h3>
                <ul id="productsList" class="list-disc mr-8 text-right"></ul>
            </div>
            
            <div>
                <h3 class="text-lg font-semibold mb-2 text-right">צעדי הגנה מומלצים:</h3>
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
            
            title.textContent = `פלייבוק הגנה - ${id}`;
            description.textContent = playbook.description || "אין תיאור";
            
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
                li.textContent = "לא זוהו מוצרים ספציפיים";
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
</html>'''

@app.route('/')
def index():
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
    
    return render_template_string(hebrew_template, 
                                 vulnerabilities=vulnerabilities,
                                 critical_count=critical_count,
                                 high_count=high_count,
                                 generate_playbook=generate_playbook)

if __name__ == '__main__':
    app.run(debug=True)