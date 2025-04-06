from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, render_template_string
import re
import bcrypt
from flask_pymongo import PyMongo
import requests
from bson import ObjectId
import google.generativeai as genai
from datetime import datetime
from markdown2 import Markdown
import os
from groq import Groq
import json
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler


app = Flask(__name__)
markdowner = Markdown()

#Database
app.config["MONGO_URI"] = "mongodb://localhost:27017/PBSC-Ignite-db"  # Update with your MongoDB URI
mongo = PyMongo(app)

# Configure Gemini API'
GENMI_API_KEY = "AIzaSyD9oeJ_5I26O4SX4v2KjNwNtMElj-dYvBM"
#GENMI_API_KEY = "AIzaSyBVL73UIy3dhEyP3OC4gCgMOQFgX2v6G8E"  # Replace with your Gemini API Key
genai.configure(api_key=GENMI_API_KEY)

# Set a secret key for session management and flashing messages
app.secret_key = 'AF011-FFAI'

@app.route('/get_notifications')
def get_notifications():
    if "user_id" not in session:
        return jsonify([])
    
    notifications = list(mongo.db.notifications.find(
        {"user_id": session["user_id"], "read": False}
    ).sort("created_at", -1).limit(5))
    
    for n in notifications:
        n["_id"] = str(n["_id"])
        
    return jsonify(notifications)

def check_daily_tasks():
   current_date = datetime.now()
   users = mongo.db.users.find({"active_modules": {"$exists": True}})
   
   for user in users:
       active_modules = sorted(user.get("active_modules", []), 
                             key=lambda x: int(x["phase_id"]))
       
       current_module = next((m for m in active_modules 
                            if m["status"] != "completed"), None)
       
       if current_module:
           start_date = current_module["start_date"]
           days_since_start = (current_date - start_date).days
           completed_tasks = sum(1 for week in current_module["learning_plan"]["weekly_schedule"]
                               for task in week["daily_tasks"]
                               if task.get("completed"))
           expected_tasks = days_since_start + 1
           
           notification = {
               "user_id": user["user_id"],
               "phase_id": current_module["phase_id"],
               "message": f"Module {current_module['name']}: {completed_tasks} of {expected_tasks} tasks completed",
               "completed": completed_tasks >= expected_tasks,
               "created_at": current_date,
               "read": False,
               "priority": "high" if completed_tasks < expected_tasks else "normal"
           }
           
           if completed_tasks < expected_tasks:
               notification["message"] += f" (Behind by {expected_tasks - completed_tasks} days)"
           
           mongo.db.notifications.insert_one(notification)

           # Check next module if current is completed
           if current_module["status"] == "completed":
               next_module = next((m for m in active_modules[active_modules.index(current_module) + 1:]), None)
               if next_module:
                   mongo.db.notifications.insert_one({
                       "user_id": user["user_id"],
                       "message": f"Ready to start Module {next_module['name']}",
                       "created_at": current_date,
                       "read": False,
                       "priority": "info"
                   })

# Initialize scheduler
#scheduler = BackgroundScheduler()
#scheduler.add_job(check_daily_tasks, 'cron', hour=7)  # Runs daily at 9 AM
#scheduler.start()
check_daily_tasks()


@app.route("/news-articles")
def news_article():
    if "user_id" in session:
        # Categories for the top section
        categories = [
            "Blockchain", "JavaScript", "Education", "Coding", "Books", "Web Development",
            "Marketing", "Deep Learning", "Social Media", "Software Development",
            "Artificial Intelligence", "Culture", "React", "UX", "Software Engineering",
            "Design", "Science", "Health", "Python", "Productivity", "Machine Learning",
            "Writing", "Self Improvement", "Technology", "Data Science", "Programming"
        ]

        # Get query parameters for topic and pagination
        query = request.args.get("q", "technology")
        page = int(request.args.get("page", 0))

        # API Request
        url = "https://medium16.p.rapidapi.com/search/stories"
        headers = {
            #"x-rapidapi-key": "2e1d6d9429msh47bb7452e4880d6p19db56jsn447674e01b3c",
            "x-rapidapi-key": "a9d206afa9msh1a3192fce899677p15fbaajsn6ccdd156cb0e",
            "x-rapidapi-host": "medium16.p.rapidapi.com",
        }
        querystring = {"q": query, "limit": "10", "page": str(page)}
        response = requests.get(url, headers=headers, params=querystring)

        # Extract stories
        stories = response.json().get("data", []) if response.status_code == 200 else []

        return render_template(
            "news_articles.html", stories=stories, query=query, page=page, categories=categories
        )
    else:
            return redirect(url_for("sign_in"))
    

# GitHub Integration
def fetch_github_projects(github_username):
    """Fetch the user's public GitHub repositories using the stored GitHub username."""
    github_api_url = f"https://api.github.com/users/{github_username}/repos"
    try:
        response = requests.get(github_api_url)
        response.raise_for_status()  # Raise error for bad responses
        repos = response.json()
        projects = [{"title": repo["name"], "description": repo["description"] or "No description available"} for repo in repos]
        return projects
    except requests.RequestException as e:
        return f"Error fetching GitHub data: {e}"

# GitHub Integration
def fetch_github_projects(github_username):
    """Fetch the user's public GitHub repositories using the stored GitHub username."""
    github_api_url = f"https://api.github.com/users/{github_username}/repos"
    try:
        response = requests.get(github_api_url)
        response.raise_for_status()  # Raise error for bad responses
        repos = response.json()
        projects = [{"title": repo["name"], "description": repo["description"] or "No description available"} for repo in repos]
        return projects
    except requests.RequestException as e:
        return f"Error fetching GitHub data: {e}"

def generate_prompt(user_data, user_query, chat_history):
    """Generate a conversational prompt including chat history and GitHub projects."""
    
    # Extract user data
    name = f"{user_data.get('firstName', 'there')} {user_data.get('lastName', '')}".strip()
    headline = user_data.get("headline", "No headline available")
    summary = user_data.get("summary", "No summary available")
    certifications = user_data.get("certifications", [])
    skills = user_data.get("skills", [])
    projects = user_data.get("projects", {}).get("items", [])
    honors = user_data.get("honors", [])
    geo = user_data.get("geo", {}).get("full", "Location not specified")
    github_username = user_data.get("github_username", None)  # Get GitHub username from user data
    
    # Process certifications
    certifications_str = []
    for cert in certifications:
        cert_name = cert.get("name", "Unnamed Certification")
        cert_authority = cert.get("authority", "Unknown Authority")
        cert_company = cert.get("company", {}).get("name", "Unknown Company")
        cert_time = f"{cert.get('start', {}).get('year', 'Unknown Start Year')} - {cert.get('end', {}).get('year', 'Unknown End Year')}"
        certifications_str.append(f"{cert_name} ({cert_authority} - {cert_company} - {cert_time})")
    
    certifications_str = ', '.join(certifications_str) if certifications_str else 'None'
    
    # Process honors
    honors_str = []
    for honor in honors:
        honor_title = honor.get("title", "Unknown Honor")
        honor_description = honor.get("description", "No description available")
        honor_issuer = honor.get("issuer", "Unknown Issuer")
        honor_time = f"{honor.get('issuedOn', {}).get('year', 'Unknown Year')}"
        honors_str.append(f"{honor_title} - {honor_description} (Issued by {honor_issuer} in {honor_time})")
    
    honors_str = ', '.join(honors_str) if honors_str else 'None'
    
    # Handle skills to ensure they are strings
    skills_str = ', '.join([skill.get("name", "Unknown Skill") for skill in skills]) if skills else 'None'

    # Handle projects to ensure they are strings
    projects_str = ', '.join([proj.get("title", "Unknown Project") for proj in projects]) if projects else 'None'

    # Fetch GitHub projects if a GitHub username exists
    github_projects_str = "None"
    if github_username:
        github_projects = fetch_github_projects(github_username)
        github_projects_str = ', '.join([f"{proj['title']}: {proj['description']}" for proj in github_projects]) if isinstance(github_projects, list) else github_projects

    # Build the history string, ensuring each entry is valid
    history_str = "\n".join([f"User: {entry.get('query', '')}\nBot: {entry.get('response', '')}" 
                            for entry in chat_history if isinstance(entry, dict) and 'query' in entry and 'response' in entry])

    # Create the new prompt
    prompt = f"""
    You are a helpful assistant responding in a friendly and conversational tone. 
    The user's profile contains the following:
    - Name: {name}
    - Headline: {headline}
    - Summary: {summary}
    - Location: {geo}
    - Certifications: {certifications_str if certifications else 'None'}
    - Skills: {skills_str if skills else 'None'}
    - Projects (LinkedIn): {projects_str if projects else 'None'}
    - Honors: {honors_str if honors else 'None'}

    Chat history:
    {history_str}

    User Question: "{user_query}"

    Respond in a friendly and concise manner:
    - Continue the conversation based on the chat history and user profile.
    - Keep responses brief and focused while maintaining context.
    """
    return prompt


# Get response from Gemini LLM
def get_gemini_response(prompt, tokens = 8192):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config={
                "temperature": 0.5,  # Lower temperature for deterministic responses
                "top_p": 0.95,
                "max_output_tokens": tokens,  # Limit response length
            },
        )
        convo = model.start_chat(history=[])
        response = convo.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

def fetch_and_save_linkedin_profile(linkedin_url=None):
    """
    Fetches the LinkedIn profile data using the LinkedIn Data API and saves it to MongoDB.

    Parameters:
    linkedin_url (str): The URL of the LinkedIn profile to fetch data from.

    Returns:
    str: A message indicating the success or failure of the operation.
    """
    # LinkedIn API request
    students_collection = mongo.db["linkedin_data"]        
    url = "https://linkedin-data-api.p.rapidapi.com/get-profile-data-by-url"

    #querystring = {"url":"https://www.linkedin.com/in/abdulfaheem011/"}

    print("LinkedIn URL - ", linkedin_url)
    querystring = {"url":linkedin_url}

    headers = {
        #"x-rapidapi-key": "2e1d6d9429msh47bb7452e4880d6p19db56jsn447674e01b3c",
        #"x-rapidapi-key": "0e567dff3cmsh89a7c64ae4f064cp1597cfjsn80a01991ebba",  # Replace with your API key
        #"x-rapidapi-key": "f7bfaf31eemsh40d3e4d40afd75dp1ab6c7jsn771476f0b98f",        
        'x-rapidapi-key': "9ac9ae8123mshf8493c7d86c01bap1d40a5jsn33fb267e423a",
        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com"
    }    

    try:
        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code == 200:
            profile_data = response.json()

            # Add user_id to profile data
            profile_data["user_id"] = session.get("user_id")

            if not profile_data["user_id"]:
                return "Failed: Missing user ID in session."

            # Check if a profile with the same user_id already exists
            existing_profile = students_collection.find_one({"user_id": session["user_id"]})

            if existing_profile:
                print("Hello")
                # If an existing profile is found, update it
                result = students_collection.update_one(
                    {"user_id": session["user_id"]},  # Match by user_id
                    {"$set": profile_data}             # Update fields with new profile data
                )
                return f"Profile data updated for user_id: {session['user_id']}."
            else:                
                # If no existing profile is found, insert the new profile data
                students_collection.insert_one(profile_data)
                return f"Profile data saved for user_id: {session['user_id']}."
        else:
            return f"Failed to retrieve data: {response.status_code}, {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/your-career_coach-leo011', methods=['POST', 'GET'])
def career_coach():
    if "user_id" in session:
        leo_chat_history = mongo.db.career_coach
        students_collection = mongo.db["linkedin_data"]
        user_collection = mongo.db.users
        
        if request.method == 'POST':
            user_id = session['user_id']
            user_query = request.form['userQuery']
            
            user_record = user_collection.find_one({'user_id': user_id})  # Assuming '_id' is the field storing user_id
            linkedin_url = user_record.get('linkedinProfile') if user_record else None
            print("Calling - LinkedIn URL - ", linkedin_url)
            
            # Fetch user data from MongoDB
            if linkedin_url:
                fetch_and_save_linkedin_profile(linkedin_url)  # Pass the LinkedIn URL as an argument
            else:
                return "LinkedIn profile URL not found", 404
            user_data = students_collection.find_one({"user_id": user_id})
            if not user_data:
                return "User data not found", 404

            # Generate prompt and get response
            prompt = generate_prompt(user_data, user_query, leo_chat_history.find({"user_id": user_id}))
            response = get_gemini_response(prompt, 300)
            
            # Convert markdown response to HTML
            html_response = markdowner.convert(response)

            existing_conversation = leo_chat_history.find_one({"user_id": user_id})

            if not existing_conversation:
                conversation_id = f"conv_{str(datetime.now().timestamp()).replace('.', '')}"
                new_conversation = {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "messages": [
                        {
                            "prompt": user_query,
                            "response": html_response,  # Store HTML version
                            "raw_response": response,  # Store original markdown
                            "time": datetime.utcnow(),
                        },
                    ]
                }
                leo_chat_history.insert_one(new_conversation)
                updated_messages = new_conversation["messages"]
            else:
                updated_messages = sorted(
                    [
                        {
                            "prompt": msg["prompt"], 
                            "response": msg.get("response", markdowner.convert(msg.get("raw_response", ""))),
                            "time": msg["time"]
                        }
                        for msg in existing_conversation["messages"]
                    ] + [{
                        "prompt": user_query, 
                        "response": html_response,
                        "raw_response": response,
                        "time": datetime.utcnow()
                    }],
                    key=lambda x: x["time"]
                )

                leo_chat_history.update_one(
                    {"user_id": user_id},
                    {"$set": {"messages": updated_messages}}
                )

            return render_template(
                "career_coach.html", 
                messages=updated_messages
            )

        # GET request: load existing conversation
        existing_conversation = leo_chat_history.find_one({"user_id": session["user_id"]})
        if existing_conversation:
            messages = [{
                "prompt": msg["prompt"],
                "response": msg.get("response", markdowner.convert(msg.get("raw_response", ""))),
                "time": msg["time"]
            } for msg in existing_conversation["messages"]]
        else:
            messages = []
            
        return render_template("career_coach.html", messages=messages)
    
    return redirect(url_for("sign_in"))


def get_roadmap_from_groq(topic):

    os.environ["GROQ_API_KEY"] = "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"
    client = Groq()
    
    prompt = f"""Create a structured learning roadmap for {topic} in this exact JSON format:
    {{
        "phases": [
            {{
                "name": "Phase Name",
                "duration": "X-Y months",
                "description": "Brief description",
                "skills": ["skill1", "skill2", "skill3"],
                "resources": {{
                    "Category1": ["Resource1", "Resource2"],
                    "Category2": ["Resource3", "Resource4"],
                    "Category3": ["Resource5", "Resource6"]
                }}
            }}
        ]
    }}
    Include exactly 4 phases."""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a technical expert. Respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=2000
    )
    
    content = response.choices[0].message.content.strip()
    return json.loads(content)

@app.route('/road-map')
def road_map():
    if "user_id" not in session:
        return redirect(url_for("sign_in"))

    user = mongo.db.users.find_one({"user_id": session['user_id']})
    if not user:
        flash("Profile not found")
        return redirect(url_for("sign_in"))

    try:
        roadmap_data = json.loads(user["road_map"])
        active_modules = user.get("active_modules", [])

        # Add progress data to roadmap phases
        for phase in roadmap_data["phases"]:
            active_module = next((m for m in active_modules if m["phase_id"] == str(roadmap_data["phases"].index(phase))), None)
            if active_module:
                phase["learning_plan"] = active_module["learning_plan"]
                phase["progress"] = active_module.get("progress", 0)
                phase["status"] = active_module.get("status", "not_started")

        return render_template('road_map.html', user=user, roadmap_data=roadmap_data)
        
    except Exception as e:
        flash(f"Error loading roadmap: {str(e)}")
        return redirect(url_for("student_profile"))
    

@app.route('/generate-plan/<phase_id>', methods=['POST'])
def generate_plan(phase_id):
    try:
        user = mongo.db.users.find_one({"user_id": session["user_id"]})
        
        # Check if plan exists
        existing_plan = next((m for m in user.get("active_modules", []) 
                            if m["phase_id"] == phase_id), None)
        if existing_plan:
            return jsonify({"status": "exists"})
        
        # Generate new plan only if it doesn't exist
        roadmap = json.loads(user["road_map"])
        phase = roadmap["phases"][int(phase_id)]
        
        os.environ["GROQ_API_KEY"] = "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"
        client = Groq()
        prompt = f"""Generate learning plan for {phase['name']} phase with skills: {', '.join(phase['skills'])}. Return pure JSON:
                {{
                    "weekly_schedule": [
                        {{
                            "week": 1,
                            "learning_objectives": ["Objective 1", "Objective 2"],
                            "daily_tasks": [
                                {{
                                    "day": 1,
                                    "tasks": ["Task 1", "Task 2"],
                                    "resources": ["Resource 1"],
                                    "duration_hours": 2
                                }}
                            ],
                            "assessment": "Project description"
                        }}
                    ]
                }}"""

        response = client.chat.completions.create(
            messages=[{"role": "system", "content": "Return valid JSON only"},
                     {"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        
        learning_plan = json.loads(response.choices[0].message.content.strip()
                                 .replace("```json", "").replace("```", ""))

        milestone = {
            "phase_id": phase_id,
            "name": phase["name"],
            "learning_plan": learning_plan,
            "start_date": datetime.now(),
            "status": "in_progress",
            "progress": 0
        }

        mongo.db.users.update_one(
            {"user_id": session["user_id"]},
            {"$addToSet": {"active_modules": milestone}}
        )
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
@app.route('/learning-plan/<phase_id>')
def learning_plan(phase_id):
   if "user_id" not in session:
       return redirect(url_for("sign_in"))
   
   user = mongo.db.users.find_one({"user_id": session["user_id"]})
   active_modules = user.get("active_modules", [])
   print(active_modules)
   current_module = next((m for m in active_modules if m["phase_id"] == phase_id), None)
   
   if not current_module:
       flash("Module not found")
       return redirect(url_for("road_map"))
   
   return render_template("learning_plan.html", module=current_module)

@app.route('/update_task_status/<phase_id>/<week_num>/<day_num>', methods=['POST'])
def update_task_status(phase_id, week_num, day_num):
   try:
       data = request.json
       user_id = session["user_id"]
       current_time = datetime.now()

       # Update task completion status and completion date
       result = mongo.db.users.update_one(
           {
               "user_id": user_id,
               "active_modules.phase_id": phase_id,
           },
           {
               "$set": {
                   "active_modules.$.learning_plan.weekly_schedule.$[week].daily_tasks.$[day].completed": data['completed'],
                   "active_modules.$.learning_plan.weekly_schedule.$[week].daily_tasks.$[day].completed_date": current_time if data['completed'] else None
               }
           },
           array_filters=[
               {"week.week": int(week_num)},
               {"day.day": int(day_num)}
           ]
       )

       # Add notification
       notification = {
           "user_id": user_id,
           "phase_id": phase_id,
           "week": int(week_num),
           "day": int(day_num),
           "message": f"Day {day_num} tasks {'completed' if data['completed'] else 'pending'} for Week {week_num}",
           "completed": data['completed'],
           "created_at": current_time,
           "read": False
       }
       mongo.db.notifications.insert_one(notification)

       # Calculate progress
       user = mongo.db.users.find_one({"user_id": user_id})
       module = next((m for m in user.get("active_modules", []) if m["phase_id"] == phase_id), None)
       
       if module:
           total_days = sum(len(week.get("daily_tasks", [])) for week in module["learning_plan"]["weekly_schedule"])
           completed_days = sum(sum(1 for task in week.get("daily_tasks", []) if task.get("completed", False)) 
                              for week in module["learning_plan"]["weekly_schedule"])
           
           progress = round((completed_days / total_days * 100)) if total_days > 0 else 0

           mongo.db.users.update_one(
               {
                   "user_id": user_id,
                   "active_modules.phase_id": phase_id
               },
               {
                   "$set": {
                       "active_modules.$.progress": progress,
                       "active_modules.$.status": "completed" if progress == 100 else "in_progress"
                   }
               }
           )

       return jsonify({"status": "success", "progress": progress})

   except Exception as e:
       return jsonify({"status": "error", "message": str(e)})

@app.route('/student_profile', methods=['POST', 'GET'])
def student_profile():
    if "user_id" not in session:
        return redirect(url_for("sign_in"))

    user_collection = mongo.db.users
    linkedin_data_collection = mongo.db.linkedin_data

    if request.method == 'POST':
        user_id = session['user_id']
        existing_profile = user_collection.find_one({"user_id": user_id})
        if not existing_profile:
            return "Profile not found", 404

        # Check if key fields were updated
        key_fields = ['career_goal', 'dream_company', 'personal_statement', 'company_preference']
        key_fields_updated = any(
            request.form.get(field, '').strip() != existing_profile.get(field, '')
            for field in key_fields
        )

        updated_profile = {
            key: value.strip() if value.strip() else existing_profile.get(key, '')
            for key, value in request.form.items() if key != 'user_id'
        }

        # Fetch LinkedIn data
        linkedin_profile = existing_profile.get('linkedinProfile')
        if linkedin_profile:
            linkedin_data = linkedin_data_collection.find_one({"user_id": user_id})
            if linkedin_data:
                updated_profile['linkedin_data'] = {
                    "name": f"{linkedin_data.get('firstName', '')} {linkedin_data.get('lastName', '')}",
                    "headline": linkedin_data.get("headline", ""),
                    "summary": linkedin_data.get("summary", ""),
                    "skills": linkedin_data.get("skills", []),
                    "educations": linkedin_data.get("educations", []),
                    "positions": linkedin_data.get("fullPositions", []),
                    "certifications": linkedin_data.get("certifications", []),
                    "languages": linkedin_data.get("languages", []),
                }

        try:
            desired_role = updated_profile.get('career_goal', 'Software Developer')
            roadmap_data = get_roadmap_from_groq(desired_role)
            updated_profile['road_map'] = json.dumps(roadmap_data)

            update_operation = {"$set": updated_profile}
            if key_fields_updated:
                update_operation["$unset"] = {"active_modules": ""}

            user_collection.update_one(
                {"user_id": user_id}, 
                update_operation
            )
            return redirect(url_for('student_profile'))

        except Exception as e:
            print(f"Error generating roadmap: {str(e)}")
            flash("Error updating profile. Please try again.")
            return redirect(url_for('student_profile'))

    profile = user_collection.find_one({"user_id": session['user_id']}) or {}
    return render_template('student_profile.html', profile=profile, user=profile)

'''
@app.route('/student_profile', methods=['POST', 'GET'])
def student_profile():
    if "user_id" not in session:
        return redirect(url_for("sign_in"))

    user_collection = mongo.db.users
    linkedin_data_collection = mongo.db.linkedin_data

    if request.method == 'POST':
        user_id = session['user_id']
        existing_profile = user_collection.find_one({"user_id": user_id})
        if not existing_profile:
            return "Profile not found", 404

        updated_profile = {
            key: value.strip() if value.strip() else existing_profile.get(key, '')
            for key, value in request.form.items() if key != 'user_id'
        }

        # Fetch LinkedIn data
        linkedin_profile = existing_profile.get('linkedinProfile')
        if linkedin_profile:
            linkedin_data = linkedin_data_collection.find_one({"user_id": user_id})
            if linkedin_data:
                updated_profile['linkedin_data'] = {
                    "name": f"{linkedin_data.get('firstName', '')} {linkedin_data.get('lastName', '')}",
                    "headline": linkedin_data.get("headline", ""),
                    "summary": linkedin_data.get("summary", ""),
                    "skills": linkedin_data.get("skills", []),
                    "educations": linkedin_data.get("educations", []),
                    "positions": linkedin_data.get("fullPositions", []),
                    "certifications": linkedin_data.get("certifications", []),
                    "languages": linkedin_data.get("languages", []),
                }

        try:
            # Generate roadmap using Groq
            desired_role = updated_profile.get('career_goal', 'Software Developer')
            print(desired_role)
            roadmap_data = get_roadmap_from_groq(desired_role)
            print(roadmap_data)
            updated_profile['road_map'] = json.dumps(roadmap_data)

            user_collection.update_one(
                {"user_id": user_id}, 
                {"$set": updated_profile,
                 "$unset": {"active_modules": ""}
                }
            )
            return redirect(url_for('student_profile'))

        except Exception as e:
            print(f"Error generating roadmap: {str(e)}")
            flash("Error updating profile. Please try again.")
            return redirect(url_for('student_profile'))

    # GET request
    profile = user_collection.find_one({"user_id": session['user_id']}) or {}
    return render_template('student_profile.html', profile=profile, user=profile)
'''
@app.route("/")
def home():
    if "user_id" in session:
        # Categories for the top section
        
        companies_collection = mongo.db.companies

        categories = [
            "Blockchain", "JavaScript", "Education", "Coding", "Books", "Web Development",
            "Marketing", "Deep Learning", "Social Media", "Software Development",
            "Artificial Intelligence", "Culture", "React", "UX", "Software Engineering",
            "Design", "Science", "Health", "Python", "Productivity", "Machine Learning",
            "Writing", "Self Improvement", "Technology", "Data Science", "Programming"
        ]

        # Fetch articles from the /news-articles route
        query = "technology"  # You can modify this based on user input or specific category
        page = 0  # Start from page 0 for simplicity
        url = "https://medium16.p.rapidapi.com/search/stories"
        headers = {
            #"x-rapidapi-key": "2e1d6d9429msh47bb7452e4880d6p19db56jsn447674e01b3c",
            "x-rapidapi-key": "a9d206afa9msh1a3192fce899677p15fbaajsn6ccdd156cb0e",
            "x-rapidapi-host": "medium16.p.rapidapi.com",
        }
        querystring = {"q": query, "limit": "5", "page": str(page)}  # Limit articles for the carousel
        current_time = datetime.now()

        companies = companies_collection.find().sort("visit_date", 1).limit(5)



        # Convert to list and pass to template
        selected_companies = list(companies)
        print(selected_companies)

        response = requests.get(url, headers=headers, params=querystring)

        articles = response.json().get("data", []) if response.status_code == 200 else []

        return render_template(
            "home.html", categories=categories, stories=articles, companies=selected_companies
        )
    else:
        return redirect(url_for("sign_in"))
    
@app.route('/mentorship')
def mentorship():
    if "user_id" in session:
        return render_template('mentorship.html')
    else:
        return redirect(url_for("sign_in"))

@app.route('/about')
def about():
    if "user_id" in session:
        return render_template('about.html')
    else:
        return redirect(url_for("sign_in"))



# Account creation
@app.route("/sign_up", methods=['POST', 'GET'])
def sign_up():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        dob = request.form['dob']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Check if passwords match
        if password != confirm_password:
            error = "Passwords do not match. Please try again."
            return render_template("sign_up.html", error=error)
        
        # Hash the password for security        
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        
        # Connect to MongoDB and check if the email or username already exists
        user_collection = mongo.db.users  # Assuming you have a 'users' collection
        #student_profile_collection = mongo.db.student_profile
        
        existing_user = user_collection.find_one({"$or": [{"email": email.lower()}, {"user_id": username.lower()}]})
        
        if existing_user:
            error = "Email or username already exists. Please use a different one."
            return render_template("sign_up.html", error=error)
        
        # Insert the new user into the MongoDB collection

        new_user = {
            "user_id": username.lower(),
            "name": name,
            "phone": phone,
            "dob": dob,
            "email": email.lower(),
            "password": hashed_password,
            "joining_date": request.form.get("startdate", None),
            "career_goal": request.form.get("career_goal", None),
            "entrepreneurship_interest": request.form.get("entrepreneurship_interest", None),
            "key_interests": request.form.get("interested_industries", "").split(", "),  # Split into list
            "dream_company": request.form.get("dream_company", None),
            "company_preference": request.form.get("company_preference", None),
            "preferred_company": request.form.get("preferred_company", None),
            "personal_statement": request.form.get("personal_statement", None),
            "github_profile": request.form.get("githubProfile", None),
            "linkedin_profile": request.form.get("linkedinProfile", None),
        }


        
        try:
            user_collection.insert_one(new_user)  # Insert new user into the 'users' collection
            #student_profile_collection.insert_one(student_profile)
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for('sign_in'))
        except Exception as e:
            error = f"An error occurred: {e}"
            return render_template("sign_up.html", error=error)
    else:
        return render_template("sign_up.html")


@app.route("/sign_in", methods=['POST', 'GET'])
def sign_in():
    if request.method == 'POST':
        email_or_user_id = request.form['username']
        password = request.form['password']
        
        # Query to find the user by email or user ID from MongoDB
        users_collection = mongo.db.users
        user = users_collection.find_one({"$or": [{"email": email_or_user_id}, {"user_id": email_or_user_id}]})
        
        if user:                        
            # The stored password is already a byte string, so no need to re-encode
            stored_hashed_password = user['password'].encode()  # Ensure it's byte format
            
            # Verify the password using bcrypt
            if bcrypt.checkpw(password.encode(), stored_hashed_password):  # Hash verification
                # Store user info in the session
                session['user_id'] = user['user_id']  # Save user ID in session
                session['name'] = user['name']
                user_name = session['name']
                flash(f"Welcome back, {user['name']}!", "success")
                return redirect(url_for('home'))

            else:
                error = "Incorrect password. Please try again."
                return render_template("sign_in.html", error=error)
        else:
            error = "No account found with the provided email or user ID."
            return render_template("sign_in.html", error=error)
    else:
        return render_template("sign_in.html")
    
@app.route("/logout")
def logout():
    #session.pop("user_id", None)
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("sign_in"))


if __name__ == '__main__':
    app.run(debug=True)
