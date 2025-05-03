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
from pathlib import Path
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import nest_asyncio
from asyncio import new_event_loop, set_event_loop, get_event_loop

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

# Configure upload folder and progress folder
UPLOAD_FOLDER = 'uploads'
PROGRESS_FOLDER = 'progress'
for folder in [UPLOAD_FOLDER, PROGRESS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Global variable to store progress
progress = {
    'current': 0,
    'total': 0,
    'status': 'idle',
    'processed_companies': 0,  # Companies that meet criteria
    'analyzed_companies': 0,   # Total companies analyzed
    'current_company': '',     # Name of current company being analyzed
    'company_progress': [],    # List of companies and their analysis status
    'start_time': None,        # Start time of the analysis
    'should_stop': False       # Flag to control stopping the process
}

# Semaphore to limit concurrent API calls
SEMAPHORE = asyncio.Semaphore(5)  # Process 5 companies at a time

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

async def analyze_company_async(description, company_name, session):
    """Asynchronous version of analyze_company"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    try:
        # Update progress with current company name
        progress['current_company'] = company_name
        
        # Add company to progress tracking
        company_entry = {
            'name': company_name,
            'status': 'analyzing',
            'score': 0,
            'start_time': datetime.now().timestamp(),
            'analysis_time': 0,
            'retry_count': 0,
            'error_message': None,
            'analysis': None  # Add this to store the analysis result
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

        async with SEMAPHORE:  # Limit concurrent API calls
            for attempt in range(max_retries):
                try:
                    # Create the completion using the async client
                    response = await openai.AsyncOpenAI().chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a sales qualification expert. Be strict in your scoring - only give 8-10 to companies that are DEFINITE matches."},
                            {"role": "user", "content": prompt + description}
                        ]
                    )
                    
                    analysis = response.choices[0].message.content
                    score = extract_score(analysis)
                    
                    # Update company entry with analysis and score
                    company_entry['status'] = 'completed'
                    company_entry['analysis_time'] = round(datetime.now().timestamp() - company_entry['start_time'], 1)
                    company_entry['analysis'] = analysis
                    company_entry['score'] = score
                    
                    return analysis
                    
                except Exception as api_error:
                    company_entry['retry_count'] = attempt + 1
                    company_entry['error_message'] = str(api_error)
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for company {company_name}: {str(api_error)}")
                        await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        logger.error(f"API Error for company {company_name} after {max_retries} attempts: {str(api_error)}")
                        company_entry['status'] = 'error'
                        company_entry['analysis_time'] = round(datetime.now().timestamp() - company_entry['start_time'], 1)
                        return f"Error analyzing company: API Error - {str(api_error)}"
                
    except Exception as e:
        logger.error(f"Error in analyze_company for {company_name}: {str(e)}")
        logger.error(traceback.format_exc())
        # Update company status to error and calculate analysis time
        if 'company_entry' in locals():
            company_entry['status'] = 'error'
            company_entry['error_message'] = str(e)
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
        filename = f'matching_companies_{timestamp}.xlsx'
        excel_file = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save to Excel
        df.to_excel(excel_file, index=False)
        logger.info(f"Created Excel file: {filename}")
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
        logger.info(f"Created CSV file: {filename}")
        return filename  # Return just the filename, not the full path
    except Exception as e:
        logger.error(f"Error in create_csv_file: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def save_progress(progress_data, results, all_results):
    """Save current progress and results to files"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        progress_file = os.path.join(PROGRESS_FOLDER, f'progress_{timestamp}.json')
        results_file = os.path.join(PROGRESS_FOLDER, f'results_{timestamp}.json')
        all_results_file = os.path.join(PROGRESS_FOLDER, f'all_results_{timestamp}.json')
        
        # Save progress
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save all results
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        # Save latest progress reference
        with open(os.path.join(PROGRESS_FOLDER, 'latest_progress.txt'), 'w') as f:
            f.write(timestamp)
            
        return timestamp
    except Exception as e:
        logger.error(f"Error saving progress: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def load_latest_progress():
    """Load the latest progress and results"""
    try:
        # Read latest progress timestamp
        latest_file = os.path.join(PROGRESS_FOLDER, 'latest_progress.txt')
        if not os.path.exists(latest_file):
            return None, [], []
            
        with open(latest_file, 'r') as f:
            timestamp = f.read().strip()
            
        # Load progress and results
        progress_file = os.path.join(PROGRESS_FOLDER, f'progress_{timestamp}.json')
        results_file = os.path.join(PROGRESS_FOLDER, f'results_{timestamp}.json')
        all_results_file = os.path.join(PROGRESS_FOLDER, f'all_results_{timestamp}.json')
        
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
            
        # Check if this is from a previous session (more than 5 minutes old)
        if 'start_time' in progress_data:
            time_diff = datetime.now().timestamp() - progress_data['start_time']
            if time_diff < 300:  # Less than 5 minutes old
                return None, [], []
            
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        with open(all_results_file, 'r') as f:
            all_results = json.load(f)
            
        return progress_data, results, all_results
    except Exception as e:
        logger.error(f"Error loading progress: {str(e)}")
        logger.error(traceback.format_exc())
        return None, [], []

@app.route('/progress')
def get_progress():
    """Get the current progress of the analysis"""
    return jsonify(progress)

@app.route('/check_progress')
def check_progress():
    """Check if there's saved progress to resume from"""
    progress_data, results, all_results = load_latest_progress()
    if progress_data and progress_data['status'] == 'processing':
        # Check if this is from a previous session
        if 'start_time' in progress_data:
            time_diff = datetime.now().timestamp() - progress_data['start_time']
            if time_diff >= 300:  # More than 5 minutes old
                return jsonify({
                    'has_progress': True,
                    'current': progress_data['current'],
                    'total': progress_data['total'],
                    'analyzed': progress_data['analyzed_companies'],
                    'processed': progress_data['processed_companies']
                })
    return jsonify({'has_progress': False})

def run_async_process(df):
    """Helper function to run async process with proper event loop handling"""
    try:
        # Create a new event loop for this thread
        loop = new_event_loop()
        set_event_loop(loop)
        
        # Run the async process
        results, all_results = loop.run_until_complete(process_companies_async(df))
        
        # Save final progress
        save_progress(progress, results, all_results)
        
        # Sort results by score in descending order
        results.sort(key=lambda x: x['score'], reverse=True)
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Create Excel and CSV files
        excel_file = None
        csv_file = None
        all_csv_file = None
        
        try:
            if len(results) > 0:
                excel_file = create_excel_file(results)
                csv_file = create_csv_file(results)
                logger.info(f"Created matching companies files: Excel={excel_file}, CSV={csv_file}")
            
            if len(all_results) > 0:
                all_csv_file = create_csv_file(all_results, all_companies=True)
                logger.info(f"Created all companies file: {all_csv_file}")
        except Exception as e:
            logger.error(f"Error creating download files: {str(e)}")
            logger.error(traceback.format_exc())
        
        return results, all_results, excel_file, csv_file, all_csv_file
        
    except Exception as e:
        logger.error(f"Error in async process: {str(e)}")
        logger.error(traceback.format_exc())
        progress['status'] = 'error'
        return None, None, None, None, None
    finally:
        try:
            # Clean up the event loop
            loop.close()
        except Exception as e:
            logger.error(f"Error closing event loop: {str(e)}")

@app.route('/start_analysis')
def start_analysis():
    """Start or resume the analysis process"""
    global progress
    try:
        # Load the saved DataFrame
        session_file = os.path.join(UPLOAD_FOLDER, 'current_analysis.csv')
        if not os.path.exists(session_file):
            return jsonify({'error': 'No file found to analyze'})
            
        df = pd.read_csv(session_file)
        
        # Update progress
        progress['should_stop'] = False
        progress['status'] = 'processing'
        progress['current'] = 0
        progress['total'] = len(df)
        progress['processed_companies'] = 0
        progress['analyzed_companies'] = 0
        progress['company_progress'] = []
        progress['start_time'] = datetime.now().timestamp()
        
        # Start processing in a separate thread
        def process_in_background():
            try:
                results, all_results, excel_file, csv_file, all_csv_file = run_async_process(df)
                
                if results is not None:
                    # Update progress with file information
                    progress['excel_file'] = excel_file
                    progress['csv_file'] = csv_file
                    progress['all_csv_file'] = all_csv_file
                    progress['total_results'] = len(results)
                    progress['total_analyzed'] = len(all_results)
                    
                    logger.info(f"Created download files: Excel={excel_file}, CSV={csv_file}, All CSV={all_csv_file}")
                
            except Exception as e:
                logger.error(f"Error in background processing: {str(e)}")
                logger.error(traceback.format_exc())
                progress['status'] = 'error'
        
        # Start the background process
        import threading
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'processing',
            'total': len(df),
            'message': 'Analysis started successfully'
        })
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})

@app.route('/resume_progress')
def resume_progress():
    """Resume from paused progress"""
    progress_data, results, all_results = load_latest_progress()
    if not progress_data:
        return jsonify({'error': 'No progress found'})
        
    # Check if this is from a previous session
    if 'start_time' in progress_data:
        time_diff = datetime.now().timestamp() - progress_data['start_time']
        if time_diff < 300:  # Less than 5 minutes old
            return jsonify({'error': 'Progress is too recent'})
    
    # Update global progress
    global progress
    progress.update(progress_data)
    progress['should_stop'] = False
    progress['status'] = 'processing'
    
    # Start processing in a separate thread
    def process_in_background():
        try:
            # Load the saved DataFrame
            session_file = os.path.join(UPLOAD_FOLDER, 'current_analysis.csv')
            df = pd.read_csv(session_file)
            
            # Skip already processed companies
            df = df.iloc[progress['current']:]
            
            results, all_results, excel_file, csv_file, all_csv_file = run_async_process(df)
            
            if results is not None:
                # Update progress with file information
                progress['excel_file'] = excel_file
                progress['csv_file'] = csv_file
                progress['all_csv_file'] = all_csv_file
                progress['total_results'] = len(results)
                progress['total_analyzed'] = len(all_results)
                
                logger.info(f"Created download files: Excel={excel_file}, CSV={csv_file}, All CSV={all_csv_file}")
            
        except Exception as e:
            logger.error(f"Error in background processing: {str(e)}")
            logger.error(traceback.format_exc())
            progress['status'] = 'error'
    
    # Start the background process
    import threading
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'resumed',
        'current': progress['current'],
        'total': progress['total'],
        'analyzed': progress['analyzed_companies'],
        'processed': progress['processed_companies']
    })

@app.route('/get_current_results')
def get_current_results():
    """Get the current results for download"""
    try:
        # Load the latest results
        progress_data, results, all_results = load_latest_progress()
        
        if not results and not all_results:
            # If no results in progress files, try to get from current progress
            if progress['processed_companies'] > 0:
                # Create results from current progress
                results = []
                all_results = []
                
                # Load the current analysis file
                session_file = os.path.join(UPLOAD_FOLDER, 'current_analysis.csv')
                if os.path.exists(session_file):
                    df = pd.read_csv(session_file)
                    
                    # Process company progress
                    for company in progress['company_progress']:
                        if company['status'] == 'completed':
                            company_result = {
                                'name': company['name'],
                                'url': company.get('url', ''),
                                'analysis': company.get('analysis', ''),
                                'score': company.get('score', 0)
                            }
                            all_results.append(company_result)
                            
                            if company.get('score', 0) >= 8:
                                results.append(company_result)
        
        if not results and not all_results:
            return jsonify({'error': 'No results available'})
            
        # Create download files
        excel_file = create_excel_file(results) if results else None
        csv_file = create_csv_file(results) if results else None
        all_csv_file = create_csv_file(all_results, all_companies=True) if all_results else None
        
        # Update progress with file information
        progress['excel_file'] = excel_file
        progress['csv_file'] = csv_file
        progress['all_csv_file'] = all_csv_file
        progress['total_results'] = len(results)
        progress['total_analyzed'] = len(all_results)
        
        logger.info(f"Created download files: Excel={excel_file}, CSV={csv_file}, All CSV={all_csv_file}")
        
        return jsonify({
            'status': 'success',
            'excel_file': excel_file,
            'csv_file': csv_file,
            'all_csv_file': all_csv_file,
            'total_results': len(results),
            'total_analyzed': len(all_results)
        })
    except Exception as e:
        logger.error(f"Error getting current results: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})

@app.route('/stop_analysis')
def stop_analysis():
    """Pause the current analysis process and create download files"""
    global progress
    progress['should_stop'] = True
    progress['status'] = 'paused'  # Changed from 'stopping' to 'paused'
    
    try:
        # Load the latest results
        _, results, all_results = load_latest_progress()
        
        if results:
            # Create download files
            excel_file = create_excel_file(results)
            csv_file = create_csv_file(results)
            all_csv_file = create_csv_file(all_results, all_companies=True)
            
            # Update progress with file information
            progress['excel_file'] = excel_file
            progress['csv_file'] = csv_file
            progress['all_csv_file'] = all_csv_file
            progress['total_results'] = len(results)
            progress['total_analyzed'] = len(all_results)
            
            logger.info(f"Created download files after pausing: Excel={excel_file}, CSV={csv_file}, All CSV={all_csv_file}")
            
            return jsonify({
                'status': 'paused',  # Changed from 'stopping' to 'paused'
                'has_results': True,
                'total_results': len(results),
                'total_analyzed': len(all_results),
                'excel_file': excel_file,
                'csv_file': csv_file,
                'all_csv_file': all_csv_file
            })
    except Exception as e:
        logger.error(f"Error creating download files after pausing: {str(e)}")
        logger.error(traceback.format_exc())
    
    return jsonify({'status': 'paused', 'has_results': False})  # Changed from 'stopping' to 'paused'

async def process_companies_async(df):
    """Process companies concurrently"""
    results = []
    all_results = []
    error_count = 0
    max_errors = 50  # Maximum number of consecutive errors before stopping
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index, row in df.iterrows():
            # Check if we should stop
            if progress['should_stop']:
                logger.info("Analysis stopped by user")
                progress['status'] = 'stopped'
                break
                
            # Skip already processed companies
            if index < progress['current']:
                continue
                
            # Update progress
            progress['current'] = index + 1
            
            try:
                # Add industry information to the description for better context
                full_description = f"{row['description']}\nIndustries: {row['industryList']}" if 'industryList' in row else row['description']
                
                # Create task for company analysis
                task = asyncio.create_task(analyze_company_async(full_description, row['name'], session))
                tasks.append((task, row))
                
                # Process in batches of 5
                if len(tasks) >= 5:
                    for task, row in tasks:
                        try:
                            analysis = await task
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
                            
                            # Reset error count on successful processing
                            error_count = 0
                                
                            # Save progress every 10 companies
                            if len(all_results) % 10 == 0:
                                save_progress(progress, results, all_results)
                                
                        except Exception as e:
                            error_count += 1
                            logger.error(f"Error processing row {row['name']}: {str(e)}")
                            if error_count >= max_errors:
                                logger.error(f"Too many consecutive errors ({max_errors}). Stopping analysis.")
                                progress['status'] = 'error'
                                progress['error_message'] = f"Analysis stopped due to too many errors ({max_errors})"
                                return results, all_results
                            continue
                    
                    tasks = []
            except Exception as e:
                error_count += 1
                logger.error(f"Error creating task for row {index}: {str(e)}")
                if error_count >= max_errors:
                    logger.error(f"Too many consecutive errors ({max_errors}). Stopping analysis.")
                    progress['status'] = 'error'
                    progress['error_message'] = f"Analysis stopped due to too many errors ({max_errors})"
                    return results, all_results
                continue
        
        # Process remaining tasks
        for task, row in tasks:
            if progress['should_stop']:
                break
                
            try:
                analysis = await task
                score = extract_score(analysis)
                
                company_result = {
                    'name': row['name'],
                    'url': normalize_url(row['website']),
                    'analysis': analysis,
                    'score': score
                }
                all_results.append(company_result)
                
                if score >= 8:
                    results.append(company_result)
                    progress['processed_companies'] += 1
                
                # Reset error count on successful processing
                error_count = 0
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing remaining task for {row['name']}: {str(e)}")
                if error_count >= max_errors:
                    logger.error(f"Too many consecutive errors ({max_errors}). Stopping analysis.")
                    progress['status'] = 'error'
                    progress['error_message'] = f"Analysis stopped due to too many errors ({max_errors})"
                    return results, all_results
                continue
    
    # Update final status
    if progress['should_stop']:
        progress['status'] = 'stopped'
    elif error_count >= max_errors:
        progress['status'] = 'error'
        progress['error_message'] = f"Analysis stopped due to too many errors ({max_errors})"
    else:
        progress['status'] = 'completed'
        
        # Create files for download
        try:
            if len(results) > 0:
                excel_file = create_excel_file(results)
                csv_file = create_csv_file(results)
                all_csv_file = create_csv_file(all_results, all_companies=True)
                
                # Update progress with file information
                progress['excel_file'] = excel_file
                progress['csv_file'] = csv_file
                progress['all_csv_file'] = all_csv_file
                progress['total_results'] = len(results)
                progress['total_analyzed'] = len(all_results)
                
                logger.info(f"Created download files: Excel={excel_file}, CSV={csv_file}, All CSV={all_csv_file}")
        except Exception as e:
            logger.error(f"Error creating download files: {str(e)}")
            logger.error(traceback.format_exc())
    
    return results, all_results

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
                    
                    # Save the DataFrame to a session file
                    session_file = os.path.join(UPLOAD_FOLDER, 'current_analysis.csv')
                    df.to_csv(session_file, index=False)
                    
                    # Initialize progress
                    progress['current'] = 0
                    progress['total'] = len(df)
                    progress['status'] = 'idle'
                    progress['processed_companies'] = 0
                    progress['analyzed_companies'] = 0
                    progress['company_progress'] = []
                    progress['start_time'] = None
                    progress['should_stop'] = False
                    
                    flash('File uploaded successfully. Click "Start Analysis" to begin processing.')
                    return render_template('index.html', 
                                        file_uploaded=True,
                                        total_companies=len(df))
                    
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
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {filename}")
            flash('File not found')
            return render_template('index.html')
            
        logger.info(f"Downloading file: {filename}")
        return send_file(
            file_path,
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