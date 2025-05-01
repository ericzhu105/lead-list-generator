from flask import Flask, render_template, request, flash, send_file
import os
from dotenv import load_dotenv
import openai
import pandas as pd
import json
from datetime import datetime
import traceback
import logging

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

def analyze_company(description):
    try:
        prompt = """You are a sales qualifier for customers for Aviato.co. Analyze if this company would be a good fit based on the following criteria:
Our product: Data API for People & Company profiles, with data on work experience, email, job history, compensation, etc primarily for tech workers.
Good fit examples:
- AI Recruiting tools
- Sourcing tools (Might need to use B2B People data or company data)
- B2B companies that need data oan people and companies for enrichment 
If its not a direct fit 7/10 then dont show it
Examples of a fit companies:
Name: Mercor
URL: https://mercor.com/
COMPANY_DESCRIPTION: Mercor is a better way to hire. We're an AI-powered platform that sources, vets, and pays your next team members.
Fit Score: 7/10
Name: Moonhub
URL: https://www.moonhub.ai/
COMPANY_DESCRIPTION: Moonhub: Revolutionizing the future of work with AI agents for recruiting. Discover Stella, your AI sourcing partner designed to simplify and streamline recruiting.
Fit Score: 10/10
Name: The Swarm
URL: https://www.theswarm.com/
COMPANY_DESCRIPTION: The Swarm equips you with high-fidelity people data and industry-leading business relationship insights.
Fit Score: 10/10

Not a good fit:
- Non-software companies (bio companies, hardware companies)
- Logistics companies for in person or hardware objects
- E Commerence companies
Name: Eikon Therapeutics 
URL: https://www.eikontx.com/
COMPANY_DESCRIPTION: Advancing breakthrough therapeutics through the purposeful integration of science and engineering
Fit Score: 1/10
Name: Vanta
URL: https://www.vanta.com/
COMPANY_DESCRIPTION: Vanta automates the complex and time-consuming process of SOC 2, HIPAA, ISO 27001, PCI, and GDPR compliance certification. Automate your security monitoring in weeks instead of months.
Fit Score: 3/10

Please analyze this company description and provide:
1. A score from 1-10 of how likely they are to be a customer
2. A brief explanation of why they would or wouldn't be a good fit
3. Whether they are a 'Yes', 'Maybe', or 'No' based on the criteria

Company description to analyze: """

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sales qualification expert."},
                {"role": "user", "content": prompt + description}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in analyze_company: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error analyzing company: {str(e)}"

def extract_score(analysis):
    try:
        # Look for a score pattern like "7/10" or "Score: 7/10"
        import re
        score_match = re.search(r'(\d+)/10', analysis)
        if score_match:
            return int(score_match.group(1))
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
        excel_file = os.path.join(UPLOAD_FOLDER, f'analysis_results_{timestamp}.xlsx')
        
        # Save to Excel
        df.to_excel(excel_file, index=False)
        return excel_file
    except Exception as e:
        logger.error(f"Error in create_excel_file: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
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
                    
                    # Process each company
                    results = []
                    for index, row in df.iterrows():
                        try:
                            # Add industry information to the description for better context
                            full_description = f"{row['description']}\nIndustries: {row['industryList']}" if 'industryList' in row else row['description']
                            
                            analysis = analyze_company(full_description)
                            score = extract_score(analysis)
                            
                            # Only include companies with score >= 7
                            if score >= 7:
                                results.append({
                                    'name': row['name'],
                                    'url': row['website'],
                                    'analysis': analysis,
                                    'score': score
                                })
                        except Exception as e:
                            logger.error(f"Error processing row {index}: {str(e)}")
                            continue
                    
                    # Sort results by score in descending order
                    results.sort(key=lambda x: x['score'], reverse=True)
                    
                    # Save results to a JSON file
                    results_file = os.path.join(UPLOAD_FOLDER, 'analysis_results.json')
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # If more than 100 results, create Excel file
                    excel_file = None
                    if len(results) > 100:
                        excel_file = create_excel_file(results)
                    
                    flash(f'File processed successfully! Found {len(results)} companies with score >= 7/10.')
                    return render_template('index.html', results=results[:100], total_results=len(results), excel_file=excel_file)
                    
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
            return render_template('index.html')
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        flash('Error downloading file')
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Changed to port 8000 