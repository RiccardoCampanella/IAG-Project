import requests
from bs4 import BeautifulSoup

google_cse_api_key = "AIzaSyBSV3jz3z5JNLhgxc7eBsILkAHVRU9Mo-Q"
google_cse_cx = "9187f65ee5d384795"

def get_answer_from_google_cse(question):
    # Define the search URL using the API key and CX
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': google_cse_api_key,
        'cx': google_cse_cx,
        'q': question,
        'num': 3,  # Number of results to fetch
    }

    # Make the HTTP request
    response = requests.get(search_url, params=params)
    
    # Parse the JSON response
    results = response.json()

    if 'items' in results:
        texts = []
        for item in results['items']:
            link = item['link']
            try:
                page_text = get_page_content(link)
                texts.append(page_text)
            except Exception as e:
                print(f"Could not fetch the content for {link}: {e}")
        return "\n\n".join(texts)  # Join all the page contents
    return "No answer found."

def get_page_content(url):
    """Fetch the page content from the given URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract all text from the page
    page_text = soup.get_text(separator=' ', strip=True)
    
    
    return page_text

# Example integration with the rest of your code:
def get_answer(question):
    try:
        # Call the Google CSE function
        answer = get_answer_from_google_cse(question)
        return answer
    except Exception as e:
        print(f"Error fetching answer: {e}")
        return None
