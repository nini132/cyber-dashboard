How to Run the Software
Step 1: Download the Code
Download the software code to a folder on your computer.

Make sure you have Python installed on your computer. If not, you can download it from here: https://www.python.org/downloads/.

Step 2: Run the Software
Open the Command Prompt (Windows) or Terminal (Mac/Linux) and navigate to the folder where the code is located.

For example, if the code is on your desktop, type:

cd C:\Users\YourUsername\Desktop\project-folder
Run the software by typing the following command:


python app.py
The software will start running, and you’ll see a confirmation message like:


* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
Step 3: Access the Software
Open your browser (e.g., Chrome, Firefox, or Edge).

Type the following address in the address bar:


http://127.0.0.1:5000
You should now see the software interface and can start using it.

Step 4: Using the Software
Sign Up: Click "Register" to create a new account.

Log In: If you already have an account, click "Log In" to sign in.

Dashboard: After logging in, you’ll be taken to the dashboard where you can view vulnerabilities, add new ones, and manage your account.

Step 5: Stopping the Software
To stop the software, go back to the Command Prompt/Terminal and press CTRL + C.

Additional Tips:
If you encounter any issues or errors, make sure all required dependencies are installed. The software should install them automatically, but if not, you can run the following command:


pip install -r requirements.txt
If you’re on Windows and python is not recognized, try using py instead:


py app.py
