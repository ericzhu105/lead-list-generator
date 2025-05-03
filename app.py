from flask import Flask, render_template, request, flash, send_file, jsonify
import os
from dotenv import load_dotenv
import openai
import pandas as pd
import json
from datetime import datetime
import traceback
import logging
import re
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flashing messages

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logger.error("OpenAI API key not found in environment variables")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variable to store progress
progress = {
    'current': 0,
    'total': 0,
    'status': 'idle',
    'processed_companies': 0,  # Companies that meet criteria
    'analyzed_companies': 0,   # Total companies analyzed
    'current_company': '',     # Name of current company being analyzed
    'company_progress': [],    # List of companies and their analysis status
    'start_time': None        # Start time of the analysis
}

def normalize_url(url):
    try:
        # Remove any whitespace
        url = url.strip()
        
        # Add https:// if no protocol is specified
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Parse the URL
        parsed = urlparse(url)
        
        # Get the main domain (e.g., example.com from sub.example.com)
        domain_parts = parsed.netloc.split('.')
        if len(domain_parts) > 2:
            # Handle cases like sub.example.com
            main_domain = '.'.join(domain_parts[-2:])
        else:
            main_domain = parsed.netloc
            
        # Reconstruct the URL with the main domain
        normalized = f"{parsed.scheme}://{main_domain}"
        
        # Add path if it exists and is not just a slash
        if parsed.path and parsed.path != '/':
            normalized += parsed.path
            
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing URL {url}: {str(e)}")
        return url

def analyze_company(description, company_name):
    try:
        # Update progress with current company name
        progress['current_company'] = company_name
        
        # Add company to progress tracking
        company_entry = {
            'name': company_name,
            'status': 'analyzing',
            'score': 0,
            'start_time': datetime.now().timestamp(),
            'analysis_time': 0
        }
        progress['company_progress'].append(company_entry)
        
        # Increment analyzed companies count before GPT call
        progress['analyzed_companies'] += 1
        
        prompt = """You are a sales qualifier for customers for Aviato.co. Analyze if this company would be a good fit based on the following criteria:
Our product: Data API for People & Company profiles, with data on work experience, email, job history, compensation, etc primarily for tech workers.

DEFINITE YES - Score 8-10:
- AI Recruiting tools that need people data
- Sourcing tools that need B2B People data
- B2B companies that need data on people and companies for enrichment
- Companies that explicitly work with tech worker data

Examples of DEFINITE YES companies:
Name: Mercor
URL: https://mercor.com/
COMPANY_DESCRIPTION: Mercor is a better way to hire. We're an AI-powered platform that sources, vets, and pays your next team members.
Fit Score: 8/10
Reason: Direct need for people data in recruiting

Name: Moonhub
URL: https://www.moonhub.ai/
COMPANY_DESCRIPTION: Moonhub: Revolutionizing the future of work with AI agents for recruiting. Discover Stella, your AI sourcing partner designed to simplify and streamline recruiting.
Fit Score: 10/10
Reason: Direct need for people data in AI recruiting

Name: The Swarm
URL: https://www.theswarm.com/
COMPANY_DESCRIPTION: The Swarm equips you with high-fidelity people data and industry-leading business relationship insights.
Fit Score: 10/10
Reason: Direct need for people data and relationship insights

NOT A FIT - Score 1-7:
- Non-software companies (bio companies, hardware companies)
- Logistics companies for in person or hardware objects
- E Commerce companies
- Companies that don't explicitly need people data
- Companies that might use the data but don't have a clear use case

Examples of NOT A FIT:
Name: Eikon Therapeutics 
URL: https://www.eikontx.com/
COMPANY_DESCRIPTION: Advancing breakthrough therapeutics through the purposeful integration of science and engineering
Fit Score: 1/10
Reason: No need for people data

Name: Vanta
URL: https://www.vanta.com/
COMPANY_DESCRIPTION: Vanta automates the complex and time-consuming process of SOC 2, HIPAA, ISO 27001, PCI, and GDPR compliance certification. Automate your security monitoring in weeks instead of months.
Fit Score: 3/10
Reason: No direct need for people data

Please analyze this company description and provide:
1. A score from 1-10 (ONLY give 8-10 if it's a DEFINITE YES)
2. A brief explanation of why they would or wouldn't be a good fit
3. A clear "Yes" or "No" verdict (no maybes)

Company description to analyze: """

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sales qualification expert. Be strict in your scoring - only give 8-10 to companies that are DEFINITE matches."},
                {"role": "user", "content": prompt + description}
            ]
        )
        
        # Update company status to completed and calculate analysis time
        company_entry['status'] = 'completed'
        company_entry['analysis_time'] = round(datetime.now().timestamp() - company_entry['start_time'], 1)
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in analyze_company: {str(e)}")
        logger.error(traceback.format_exc())
        # Update company status to error and calculate analysis time
        company_entry['status'] = 'error'
        company_entry['analysis_time'] = round(datetime.now().timestamp() - company_entry['start_time'], 1)
        return f"Error analyzing company: {str(e)}"

def extract_score(analysis):
    try:
        # Look for a score pattern like "7/10" or "Score: 7/10"
        score_match = re.search(r'(\d+)/10', analysis)
        if score_match:
            score = int(score_match.group(1))
            # Only return scores of 8 or higher
            return score if score >= 8 else 0
        return 0
    except Exception as e:
        logger.error(f"Error in extract_score: {str(e)}")
        return 0

def create_excel_file(results):
    try:
        # Create a DataFrame from the results
        df = pd.DataFrame(results)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'analysis_results_{timestamp}.xlsx'
        excel_file = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save to Excel
        df.to_excel(excel_file, index=False)
        return filename  # Return just the filename, not the full path
    except Exception as e:
        logger.error(f"Error in create_excel_file: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_csv_file(results, all_companies=False):
    try:
        # Create a DataFrame from the results
        df = pd.DataFrame(results)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = 'all_companies' if all_companies else 'matching_companies'
        filename = f'{prefix}_{timestamp}.csv'
        csv_file = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save to CSV
        df.to_csv(csv_file, index=False)
        return filename  # Return just the filename, not the full path
    except Exception as e:
        logger.error(f"Error in create_csv_file: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@app.route('/progress')
def get_progress():
    return jsonify(progress)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global progress
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file selected')
                return render_template('index.html')
            
            file = request.files['file']
            
            if file.filename == '':
                flash('No file selected')
                return render_template('index.html')
            
            if file and file.filename.endswith('.csv'):
                try:
                    # Read the CSV file
                    df = pd.read_csv(file)
                    logger.debug(f"CSV columns: {df.columns.tolist()}")
                    
                    # Check if required columns exist
                    required_columns = ['name', 'description', 'website']
                    if not all(col in df.columns for col in required_columns):
                        missing_cols = [col for col in required_columns if col not in df.columns]
                        flash(f'Missing required columns: {", ".join(missing_cols)}')
                        return render_template('index.html')
                    
                    # Initialize progress
                    progress['current'] = 0
                    progress['total'] = len(df)
                    progress['status'] = 'processing'
                    progress['processed_companies'] = 0
                    progress['analyzed_companies'] = 0
                    progress['company_progress'] = []  # Reset company progress
                    progress['start_time'] = datetime.now().timestamp()  # Set start time
                    
                    # Process each company
                    results = []
                    all_results = []  # Store all analyzed companies
                    for index, row in df.iterrows():
                        try:
                            # Update progress
                            progress['current'] = index + 1
                            
                            # Add industry information to the description for better context
                            full_description = f"{row['description']}\nIndustries: {row['industryList']}" if 'industryList' in row else row['description']
                            
                            analysis = analyze_company(full_description, row['name'])
                            score = extract_score(analysis)
                            
                            # Store all analyzed companies
                            company_result = {
                                'name': row['name'],
                                'url': normalize_url(row['website']),
                                'analysis': analysis,
                                'score': score
                            }
                            all_results.append(company_result)
                            
                            # Only include companies with score >= 8 (definite yes)
                            if score >= 8:
                                results.append(company_result)
                                progress['processed_companies'] += 1
                        except Exception as e:
                            logger.error(f"Error processing row {index}: {str(e)}")
                            continue
                    
                    # Update progress to completed
                    progress['status'] = 'completed'
                    
                    # Sort results by score in descending order
                    results.sort(key=lambda x: x['score'], reverse=True)
                    all_results.sort(key=lambda x: x['score'], reverse=True)
                    
                    # Save results to JSON files
                    results_file = os.path.join(UPLOAD_FOLDER, 'analysis_results.json')
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    all_results_file = os.path.join(UPLOAD_FOLDER, 'all_companies_results.json')
                    with open(all_results_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    
                    # Create Excel and CSV files
                    excel_file = None
                    csv_file = None
                    all_csv_file = None
                    
                    if len(results) > 0:
                        excel_file = create_excel_file(results)
                        csv_file = create_csv_file(results)
                    
                    if len(all_results) > 0:
                        all_csv_file = create_csv_file(all_results, all_companies=True)
                    
                    flash(f'File processed successfully! Found {len(results)} companies with score >= 8/10.')
                    return render_template('index.html', 
                                        results=results[:100], 
                                        total_results=len(results),
                                        total_analyzed=len(all_results),
                                        excel_file=excel_file,
                                        csv_file=csv_file,
                                        all_csv_file=all_csv_file)
                    
                except Exception as e:
                    logger.error(f"Error processing CSV: {str(e)}")
                    logger.error(traceback.format_exc())
                    flash(f'Error processing file: {str(e)}')
            else:
                flash('Please upload a CSV file')
            
            return render_template('index.html')
        except Exception as e:
            logger.error(f"Error in upload_file: {str(e)}")
            logger.error(traceback.format_exc())
            flash(f'An unexpected error occurred: {str(e)}')
            progress['status'] = 'error'
            return render_template('index.html')
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    try:
        # Ensure the filename is safe and exists in the upload folder
        if not os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
            flash('File not found')
            return render_template('index.html')
            
        return send_file(
            os.path.join(UPLOAD_FOLDER, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        logger.error(traceback.format_exc())
        flash('Error downloading file')
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)  # Changed to port 3000 