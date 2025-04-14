# simulator.py
# Suggestion: Install Faker for realistic data -> pip install Faker

import json
import uuid
import requests # Although not sending yet, keep for future use
import random
import time
from datetime import datetime, timedelta
from faker import Faker

fake = Faker() # Initialize Faker

class CRCSimulator:
    def __init__(self, crc_api_base_url=None, config_file="config.json"):
        """
        Initializes the CRC Simulator.

        Args:
            crc_api_base_url (str, optional): The base URL of the CRC API. Defaults to None.
            config_file (str, optional): Path to the configuration file. Defaults to "config.json".
        """
        self.crc_api_base_url = crc_api_base_url
        self.config = self.load_config(config_file)
        print(f"CRCSimulator initialized. Target API URL: {self.crc_api_base_url if self.crc_api_base_url else 'Not set (printing only)'}")

    def load_config(self, config_file):
        """
        Loads configuration from a JSON file.
        Provides default values if the file or specific settings are missing.
        """
        default_config = {
            "SIEM_Alert": {"default_severity": "Medium"},
            "Login_Alert": {"default_status": "Success"},
            "Smart_Fence_Alert": {"default_status": "Breached"},
            "Location_Based_Alert": {"default_user": "Unknown"},
            "Motion_Sensor_Alert": {"default_status": "Detected"},
            "IR_Sensor_Alert": {"default_status": "Detected"}
        }
        try:
            with open(config_file, "r") as f:
                loaded_config = json.load(f)
                # Merge loaded config with defaults to ensure all keys exist
                for event_type, defaults in default_config.items():
                    if event_type not in loaded_config:
                        loaded_config[event_type] = defaults
                    else:
                        for key, value in defaults.items():
                            loaded_config[event_type].setdefault(key, value)
                print(f"Configuration loaded from '{config_file}'")
                return loaded_config
        except FileNotFoundError:
            print(f"Warning: Config file '{config_file}' not found. Using default configuration.")
            return default_config
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{config_file}'. Using default configuration.")
            return default_config
        except Exception as e:
            print(f"Error loading config file '{config_file}': {e}. Using default configuration.")
            return default_config

    def save_config(self, config_file="config.json"):
        """Saves the current configuration to a JSON file."""
        try:
            with open(config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to '{config_file}'")
        except Exception as e:
            print(f"Error saving config file '{config_file}': {e}")


    def convert_to_crc_format(self, event_type, event_data):
        """Formats the event data into the expected CRC structure."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat() + "Z" # Adding Z for UTC indication common in APIs
        full_event = {
            "crcEventId": event_id,
            "crcTimestamp": timestamp,
            "crcEventType": event_type,
            "crcEventData": event_data # The specific details for this event type
        }
        return full_event

    def send_event(self, event_type, event_data):
        """Formats and 'sends' the event (currently prints it)."""
        full_event = self.convert_to_crc_format(event_type, event_data)
        print("-" * 20 + f" Event Generated ({event_type}) " + "-" * 20)
        print(json.dumps(full_event, indent=2))
        print("-" * (42 + len(event_type))) # Match the line length above

        if self.crc_api_base_url:
            url = f"{self.crc_api_base_url}/events" # Assuming an '/events' endpoint
            headers = {"Content-Type": "application/json"}
            try:
                # Note: Sending is disabled by default, enable if needed
                # response = requests.post(url, headers=headers, json=full_event, timeout=10)
                # response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                # print(f"--> Event successfully sent to CRC API: {url} (Status Code: {response.status_code})")
                print(f"--> NOTE: Sending to API ({url}) is currently disabled in the code.")
                pass # Comment out pass and uncomment requests.post above to enable sending
            except requests.exceptions.RequestException as e:
                print(f"Error sending event to CRC API ({url}): {e}")
            except Exception as e:
                 print(f"An unexpected error occurred during sending to {url}: {e}")
        else:
            print("--> API URL not set. Event printed to console only.")


    # --- Detail Generation Functions ---

    def get_siem_alert_details(self, manual=False):
        default_severity = self.config.get("SIEM_Alert", {}).get("default_severity", "Medium")
        possible_severities = ["Low", "Medium", "High", "Critical"]
        if manual:
            severity = input(f"Enter severity ({', '.join(possible_severities)}) [default: {default_severity}]: ").capitalize().strip()
            if not severity or severity not in possible_severities:
                print(f"Invalid or empty severity. Using default: {default_severity}.")
                severity = default_severity
            description = input("Enter description: ").strip() or f"Manual SIEM event on {datetime.now()}"
            user = input("Enter user: ").strip() or fake.user_name()
            ip = input("Enter source IP address: ").strip() or fake.ipv4()
            target_resource = input("Enter target resource: ").strip() or f"/api/v1/{fake.uri_path()}"
        else:
            severity = random.choice(possible_severities)
            description = fake.sentence(nb_words=10) + f" (Rule: {random.choice(['FW-Policy-Violation', 'Malware-Detected', 'Anomalous-Login', 'Data-Exfiltration'])})"
            user = fake.user_name()
            ip = fake.ipv4()
            target_resource = f"/api/v1/{fake.uri_path()}/{random.choice(['users','data','config'])}"

        return {
            "source": "SIEM",
            "alertName": f"{severity} severity alert detected",
            "severity": severity,
            "description": description,
            "affectedUser": user,
            "sourceIP": ip,
            "destinationIP": fake.ipv4(), # Add a destination IP
            "protocol": random.choice(["TCP", "UDP", "ICMP"]),
            "sourcePort": random.randint(1024, 65535),
            "destinationPort": random.choice([80, 443, 22, 3389, 53, random.randint(1024, 65535)]),
            "deviceAction": random.choice(["Allowed", "Blocked", "Logged", "Alerted"]),
            "targetResource": target_resource,
            "additionalInfo": {"rule_id": f"SIEM-{random.randint(1000,9999)}", "threat_score": round(random.uniform(0.1, 1.0), 2)}
        }

    def get_login_alert_details(self, manual=False):
        default_status = self.config.get("Login_Alert", {}).get("default_status", "Success")
        possible_statuses = ["Success", "Failure"]
        if manual:
            status = input(f"Enter status ({', '.join(possible_statuses)}) [default: {default_status}]: ").capitalize().strip()
            if not status or status not in possible_statuses:
                print(f"Invalid or empty status. Using default: {default_status}.")
                status = default_status
            user = input("Enter user: ").strip() or fake.user_name()
            ip = input("Enter source IP address: ").strip() or fake.ipv4()
            auth_method = input("Enter auth method (Password, MFA, SSO): ").strip() or "Password"
        else:
            status = random.choices(possible_statuses, weights=[85, 15], k=1)[0] # More successes
            user = fake.user_name()
            ip = fake.ipv4()
            auth_method = random.choice(["Password", "MFA", "SSO", "API Key"])

        return {
            "source": "Authentication Service",
            "loginStatus": status,
            "username": user,
            "sourceIP": ip,
            "userAgent": fake.user_agent(),
            "authenticationMethod": auth_method,
            "failureReason": fake.sentence(nb_words=5) if status == "Failure" else None,
            "loginTimestamp": (datetime.now() - timedelta(seconds=random.randint(1, 300))).isoformat() + "Z"
        }

    def get_smart_fence_alert_details(self, manual=False):
        default_status = self.config.get("Smart_Fence_Alert", {}).get("default_status", "Breached")
        possible_statuses = ["Breached", "Secure", "Tamper Detected", "Low Battery"]
        if manual:
            location = input("Enter fence location/segment ID: ").strip() or f"Segment-{random.randint(100,999)}"
            type_alert = input("Enter alert type (e.g., Climb, Cut, Tamper): ").strip() or "Climb"
            status = input(f"Enter status ({', '.join(possible_statuses)}) [default: {default_status}]: ").strip() or default_status
            if status not in possible_statuses: status = default_status
        else:
            location = f"{random.choice(['North Perimeter', 'East Gate', 'Warehouse Sector', 'Restricted Zone'])} Segment-{random.randint(100,999)}"
            type_alert = random.choice(["Climb Attempt", "Fence Cut", "Tamper Detected", "Impact Detected", "Zone Entry"])
            status = random.choices(possible_statuses, weights=[60, 30, 5, 5], k=1)[0]

        return {
            "source": "Smart Fence Controller",
            "fenceId": f"FNC-{random.randint(10,99)}",
            "segmentId": location,
            "alertType": type_alert,
            "status": status,
            "detectionTimestamp": (datetime.now() - timedelta(seconds=random.randint(1, 120))).isoformat() + "Z",
            "sensorData": {"vibration": round(random.uniform(0, 5.0), 2) if "Impact" in type_alert or "Tamper" in type_alert else 0,
                           "voltage": round(random.uniform(11.5, 12.5), 2) if status != "Low Battery" else round(random.uniform(10.0, 11.0), 2)}
        }

    def get_location_based_alert_details(self, manual=False):
        default_user = self.config.get("Location_Based_Alert", {}).get("default_user", fake.user_name())
        if manual:
            user = input(f"Enter user [default: {default_user}]: ").strip() or default_user
            location_desc = input("Enter location description (e.g., Warehouse Floor): ").strip() or "Main Office"
            latitude = input("Enter latitude: ").strip() or str(fake.latitude())
            longitude = input("Enter longitude: ").strip() or str(fake.longitude())
            event_trigger = input("Enter event trigger (e.g., Geofence Entry, Panic Button): ").strip() or "Geofence Entry"
        else:
            user = fake.user_name()
            location_desc = f"{random.choice(['Building', 'Site', 'Area'])} {fake.word().capitalize()}"
            latitude = str(fake.latitude())
            longitude = str(fake.longitude())
            event_trigger = random.choice(["Geofence Entry", "Geofence Exit", "Panic Button", "Man Down Alert", "Asset Movement"])

        return {
            "source": "Personnel Tracking System",
            "userId": user,
            "deviceId": f"DEV-{random.randint(10000, 99999)}",
            "locationDescription": location_desc,
            "latitude": latitude,
            "longitude": longitude,
            "trigger": event_trigger,
            "speed": round(random.uniform(0, 5.0), 1) if "Movement" in event_trigger else 0, # km/h
            "altitude": round(random.uniform(50, 150), 1), # meters
            "accuracy": round(random.uniform(5, 50), 1) # meters
        }

    def get_motion_sensor_alert_details(self, manual=False):
        default_status = self.config.get("Motion_Sensor_Alert", {}).get("default_status", "Detected")
        possible_statuses = ["Detected", "Clear"]
        if manual:
            location = input("Enter sensor location: ").strip() or f"Room {random.randint(101, 599)}"
            status = input(f"Enter status ({', '.join(possible_statuses)}) [default: {default_status}]: ").capitalize().strip()
            if not status or status not in possible_statuses: status = default_status
            timestamp_str = input("Enter timestamp (YYYY-MM-DDTHH:MM:SSZ): ").strip() or (datetime.now() - timedelta(seconds=random.randint(1, 180))).isoformat() + "Z"
        else:
            location = f"{random.choice(['Corridor', 'Office', 'Entrance', 'Storage'])} {random.randint(1, 50)}"
            status = random.choices(possible_statuses, weights=[70, 30], k=1)[0]
            timestamp_str = (datetime.now() - timedelta(seconds=random.randint(1, 180))).isoformat() + "Z"

        return {
            "source": "PIR Motion Sensor",
            "sensorId": f"MOT-{random.randint(1000, 9999)}",
            "location": location,
            "status": status,
            "detectionTimestamp": timestamp_str,
            "sensitivityLevel": random.choice(["Low", "Medium", "High"])
        }

    def get_ir_sensor_alert_details(self, manual=False):
        default_status = self.config.get("IR_Sensor_Alert", {}).get("default_status", "Detected")
        possible_statuses = ["Detected", "Clear", "Obscured"]
        if manual:
            location = input("Enter sensor location: ").strip() or f"Doorway {random.randint(1, 20)}"
            status = input(f"Enter status ({', '.join(possible_statuses)}) [default: {default_status}]: ").capitalize().strip()
            if not status or status not in possible_statuses: status = default_status
            timestamp_str = input("Enter timestamp (YYYY-MM-DDTHH:MM:SSZ): ").strip() or (datetime.now() - timedelta(seconds=random.randint(1, 180))).isoformat() + "Z"
        else:
            location = f"{random.choice(['Main Gate', 'Window', 'Passageway', 'Secure Entry'])} {random.randint(1, 10)}"
            status = random.choices(possible_statuses, weights=[65, 30, 5], k=1)[0]
            timestamp_str = (datetime.now() - timedelta(seconds=random.randint(1, 180))).isoformat() + "Z"

        return {
            "source": "IR Beam Sensor",
            "sensorId": f"IR-{random.randint(100, 999)}",
            "location": location,
            "status": status, # Beam broken (Detected) or clear
            "beamStatusTimestamp": timestamp_str,
            "beamStrength": round(random.uniform(70.0, 100.0), 1) if status != "Obscured" else round(random.uniform(10.0, 50.0), 1)
        }

    # --- Simulation Logic ---

    def simulate_event(self, event_type, manual=False):
        """Simulates a single event of the specified type."""
        event_data = None
        if event_type == "SIEM_Alert":
            event_data = self.get_siem_alert_details(manual)
        elif event_type == "Login_Alert":
            event_data = self.get_login_alert_details(manual)
        elif event_type == "Smart_Fence_Alert":
            event_data = self.get_smart_fence_alert_details(manual)
        elif event_type == "Location_Based_Alert":
            event_data = self.get_location_based_alert_details(manual)
        elif event_type == "Motion_Sensor_Alert":
            event_data = self.get_motion_sensor_alert_details(manual)
        elif event_type == "IR_Sensor_Alert":
            event_data = self.get_ir_sensor_alert_details(manual)
        else:
            print(f"Error: Unknown event type '{event_type}'")
            return

        if event_data:
            self.send_event(event_type, event_data)

# --- Main Execution / User Interface ---

def run_automation(simulator):
    """Runs the event simulator in automation mode."""
    while True:
        try:
            num_events_str = input("Enter the number of events to generate: ").strip()
            if not num_events_str: raise ValueError("Input cannot be empty.")
            num_events = int(num_events_str)
            if num_events <= 0: raise ValueError("Number of events must be positive.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a positive integer.")

    while True:#
        try:
            delay_str = input("Enter the delay between events in seconds (e.g., 0.5): ").strip()
            if not delay_str: raise ValueError("Input cannot be empty.")
            delay = float(delay_str)
            if delay < 0: raise ValueError("Delay cannot be negative.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a non-negative number.")

    event_mode = ""
    while event_mode not in ["random", "type"]:
        event_mode = input("Run automation on random mode or select specific event type? (random/type): ").strip().lower()
        if event_mode not in ["random", "type"]:
            print("Invalid input, please choose 'random' or 'type'.")

    specific_event_type = None
    if event_mode == "type":
        valid_event_types = get_valid_event_types()
        while True:
            print("""\nAvailable event types:""")
            for i, etype in enumerate(valid_event_types):
                print(f"{i+1}. {etype}")
            event_type_choice = input(f"Enter the number or name of the event type to simulate: ").strip()

            try:
                choice_index = int(event_type_choice) - 1
                if 0 <= choice_index < len(valid_event_types):
                    specific_event_type = valid_event_types[choice_index]
                    break
                else:
                    print("Invalid number.")
            except ValueError:
                # Check if input matches a type name (case-insensitive)
                matched_type = next((etype for etype in valid_event_types if etype.lower() == event_type_choice.lower()), None)
                if matched_type:
                    specific_event_type = matched_type
                    break
                else:
                    print("Invalid event type name.")
    print(f"\nStarting automation: Generating {num_events} event(s) with {delay}s delay...")
    print("Mode:", event_mode.capitalize() + (", Type:" + specific_event_type if specific_event_type else ""))
    print("Press Ctrl+C to stop early.")
    try:
        for i in range(num_events):
            event_to_simulate = specific_event_type if specific_event_type else random.choice(get_valid_event_types())
            print(f"\n[Automation Event {i+1}/{num_events}]")
            simulator.simulate_event(event_to_simulate, manual=False) # Automation always uses generated data#
            if delay > 0:
                time.sleep(delay)
        print("\nAutomation finished.")
    except KeyboardInterrupt:#
        print("\nAutomation stopped by user.")

def change_crc_api_url(simulator):
    """Allows the user to change the CRC API URL."""
    print(
        f"\nCurrent CRC API Base URL: {simulator.crc_api_base_url if simulator.crc_api_base_url else 'Not set (printing only)'}")
    new_url = input("Enter the new CRC API base URL (leave empty to disable sending, 'back' to cancel): ").strip()#
    if new_url.lower() == "back":
        return
    if not new_url:
        simulator.crc_api_base_url = None
        print("CRC API URL cleared. Events will only be printed.")
    else:
        # Basic validation (can be improved)
        if not new_url.startswith(("http://", "https://")):
             print("Warning: URL does not start with http:// or https://. Make sure it's correct.")#
        simulator.crc_api_base_url = new_url
        print(f"CRC API base URL updated to: {simulator.crc_api_base_url}")

def change_config(simulator):
    """Allows the user to change the default configuration settings."""
    config_changed = False
    while True:
        print("\nCurrent Default Configuration Settings:")
        valid_event_types = get_valid_event_types()
        for i, event_type in enumerate(valid_event_types):
            print(f"\n{i+1}. {event_type}:")
            settings = simulator.config.get(event_type, {})
            if settings:
                for key, value in settings.items():
                    print(f"  - {key}: {value}")
            else:
                print("  (No specific defaults)")
        type_choice = input("\nEnter the number or name of the event type to modify (or 'back' to return): ").strip()
        print(f"type_choice: {type_choice}")
        print(f"valid_event_types: {valid_event_types}")
        if type_choice.lower() == "back":
            break#

        selected_type = None
        try:
            choice_index = int(type_choice) - 1
            if 0 <= choice_index < len(valid_event_types):
                selected_type = valid_event_types[choice_index]
            else:
                print("Invalid number.")#
                continue
        except ValueError:
            matched_type = next((etype for etype in valid_event_types if etype.lower() == type_choice.lower()), None)
            if matched_type:
                selected_type = matched_type
            else:
                print("Invalid event type name.")
                continue#

        if selected_type not in simulator.config:
             print(f"No configurable defaults found for {selected_type}.")
             continue#

        print(f"\nSettings for {selected_type}:")
        available_settings = list(simulator.config[selected_type].keys())
        for j, setting_key in enumerate(available_settings):
             print(str(j+1) + ". " + setting_key + ": " + str(simulator.config[selected_type][setting_key]))

        setting_choice = input(f"Enter the number or name of the setting to modify (or 'back'): ").strip()
        if setting_choice.lower() == "back":
            continue#

        selected_setting = None
        try:
            setting_index = int(setting_choice) - 1
            if 0 <= setting_index < len(available_settings):
                selected_setting = available_settings[setting_index]
            else:
                 print("Invalid number.")#
                 continue
        except ValueError:
             matched_setting = next((skey for skey in available_settings if skey.lower() == setting_choice.lower()), None)
             if matched_setting:
                  selected_setting = matched_setting
             else:
                  print("Invalid setting name.")#
                  continue

        current_value = simulator.config[selected_type][selected_setting]
        new_value = input(f"Enter the new value for '{selected_setting}' (currently: '{current_value}'): ").strip()

        if not new_value:#
            print("Value cannot be empty. No change made.")
            continue

        # Simple type conversion attempt (can be improved)
        original_type = type(current_value)
        try:
            if original_type == bool:
                new_value = new_value.lower() in ['true', '1', 't', 'y', 'yes']
            elif original_type == int:
                new_value = int(new_value)
            elif original_type == float:
                new_value = float(new_value)
            # Add other types if needed
        except ValueError:
             print(f"Could not convert '{new_value}' to the expected type ({original_type.__name__}). Setting as string.")
             new_value = str(new_value) # Fallback to string

        simulator.config[selected_type][selected_setting] = new_value
        config_changed = True
        print(f"Setting '{selected_setting}' for '{selected_type}' updated to '{new_value}'.")#

    if config_changed:
        save_conf = input("Configuration changed. Save to config.json? (y/n): ").strip().lower()
        if save_conf == 'y':
             simulator.save_config()


def get_valid_event_types():
     """Returns a list of valid event types based on simulator methods."""
     # Dynamically find methods starting with 'get_' and ending with '_details'
     # Or keep a static list for simplicity:
     return [
         "SIEM_Alert", "Login_Alert", "Smart_Fence_Alert",
         "Location_Based_Alert", "Motion_Sensor_Alert", "IR_Sensor_Alert"
     ]

def main():
    """Main function to run the event simulator console interface."""
    # Ask for API URL at the start
    crc_api_url_input = input("Enter the CRC API base URL (e.g., http://localhost:8080, leave empty to print only): ").strip()
    simulator = CRCSimulator(crc_api_base_url=crc_api_url_input or None)

    valid_event_types = get_valid_event_types()

    while True:
        print("---" + " CRC Event Simulator ---")
        print("Options:")
        # Dynamically generate options for simulating events
        for i, event_type in enumerate(valid_event_types):
            print(f"{i+1}. Simulate {event_type}")

        print("-" * 20)
        print(str(len(valid_event_types)+1) + ". Run Automation")
        print(f"{len(valid_event_types)+2}. Change Configuration Defaults")
        print(f"{len(valid_event_types)+3}. Change CRC API URL")
        print(f"{len(valid_event_types)+4}. Exit")
        print("-" * 20)
#
        choice = input("Enter your choice: ").strip()

        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(valid_event_types):
                selected_event_type = valid_event_types[choice_num-1]
                manual_input = input(f"Simulate '{selected_event_type}' - Enter details manually? (y/n) [default: n]: ").lower().strip() == 'y'
                simulator.simulate_event(selected_event_type, manual_input)
            elif choice_num == len(valid_event_types) + 1:
                run_automation(simulator)
            elif choice_num == len(valid_event_types) + 2:
                 change_config(simulator)
            elif choice_num == len(valid_event_types) + 3:
                change_crc_api_url(simulator)
            elif choice_num == len(valid_event_types) + 4:
                print("Exiting simulator.")
                break
            else:
                print("Invalid choice number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}") # Catch other potential errors

if __name__ == "__main__":
    main()
