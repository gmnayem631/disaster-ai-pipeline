import spacy
import os
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Flood-specific keywords
FLOOD_KEYWORDS = ["flood", "flooding", "floodwater", "flooded", "inundated", "waterlogging"]

def extract_disaster_type(text):
    """Check if flood-related keywords exist"""
    text_lower = text.lower()
    found_keywords = [kw for kw in FLOOD_KEYWORDS if kw in text_lower]
    if found_keywords:
        return "flood"
    return "unknown"

def extract_categorized_numbers(text):
    """Extract numbers with context (deaths, injured, affected)"""
    deaths = []
    injured = []
    affected = []
    
    # Pattern for deaths: look for numbers near death-related words
    death_pattern = r'(\d+)\s*(?:people?|persons?|individuals?)?\s*(?:died|dead|killed|death|deaths)'
    death_matches = re.findall(death_pattern, text.lower())
    deaths.extend(death_matches)
    
    # Alternative pattern: "death toll", "died", etc. before number
    death_pattern2 = r'(?:died|dead|killed|death toll|deaths).*?(\d+)'
    death_matches2 = re.findall(death_pattern2, text.lower())
    deaths.extend(death_matches2)
    
    # Pattern for injured
    injured_pattern = r'(\d+)\s*(?:people?|persons?)?\s*(?:injured|wounded|hurt)'
    injured_matches = re.findall(injured_pattern, text.lower())
    injured.extend(injured_matches)
    
    # Pattern for affected/stranded/marooned
    affected_pattern = r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|lakh|thousand|people?|persons?|families)?\s*(?:affected|stranded|marooned|displaced|impacted)'
    affected_matches = re.findall(affected_pattern, text.lower())
    affected.extend(affected_matches)
    
    # Alternative: "affected" before number
    affected_pattern2 = r'(?:affected|stranded|marooned|displaced).*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|lakh|thousand|people?|families)?'
    affected_matches2 = re.findall(affected_pattern2, text.lower())
    affected.extend(affected_matches2)
    
    return {
        "deaths": list(set(deaths)) if deaths else ["0 or not mentioned"],
        "injured": list(set(injured)) if injured else ["not mentioned"],
        "affected": list(set(affected)) if affected else ["not mentioned"]
    }

def extract_locations_with_regex(text):
    """
    Extract locations using regex patterns for Bangladesh-specific contexts
    This compensates for spaCy's inability to recognize Bangladeshi place names
    """
    districts = []
    upazilas = []
    
    # Pattern 1: "X district" or "X zila"
    district_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:district|zila|zilla)'
    district_matches = re.findall(district_pattern, text)
    districts.extend(district_matches)
    
    # Pattern 2: "X upazila" or "X thana"
    upazila_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:upazila|upazilla|thana|sub-district)'
    upazila_matches = re.findall(upazila_pattern, text)
    upazilas.extend(upazila_matches)
    
    # Pattern 3: "upazilas -- X, Y, and Z" or "upazilas: X, Y, Z"
    list_pattern = r'upazilas?\s*(?:--|:)\s*([A-Z][^.]+?)(?:\.|,\s*(?:Nearly|Almost|About|\d))'
    list_matches = re.findall(list_pattern, text)
    for match in list_matches:
        # Split by comma and "and"
        places = re.split(r',\s*(?:and\s+)?', match)
        upazilas.extend([p.strip() for p in places if p.strip()])
    
    # Pattern 4: "districts -- X, Y, and Z"
    district_list_pattern = r'districts?\s*(?:--|:)\s*([A-Z][^.]+?)(?:\.|,\s*(?:Nearly|Almost|About|\d))'
    district_list_matches = re.findall(district_list_pattern, text)
    for match in district_list_matches:
        places = re.split(r',\s*(?:and\s+)?', match)
        districts.extend([p.strip() for p in places if p.strip()])
    
    return {
        "districts": list(set(districts)),
        "upazilas": list(set(upazilas))
    }

def categorize_locations_by_context(doc, locations):
    """
    Categorize locations based on surrounding context (district/upazila keywords)
    This handles locations that spaCy DOES recognize
    """
    districts = []
    upazilas = []
    uncertain = []
    
    for loc in locations:
        # Find this location entity in the doc and get context
        context_found = False
        
        for ent in doc.ents:
            if ent.text == loc and ent.label_ == "GPE":
                # Get surrounding words (5 tokens before and after)
                start_idx = max(0, ent.start - 5)
                end_idx = min(len(doc), ent.end + 5)
                context = doc[start_idx:end_idx].text.lower()
                
                # Check for administrative level keywords in context
                if any(keyword in context for keyword in ["upazila", "upazilla", "thana", "sub-district"]):
                    upazilas.append(loc)
                    context_found = True
                elif any(keyword in context for keyword in ["district", "zila", "zilla"]):
                    districts.append(loc)
                    context_found = True
                
                break
        
        # If no context clues found, mark as uncertain
        if not context_found:
            uncertain.append(loc)
    
    return {
        "districts": districts,
        "upazilas": upazilas,
        "uncertain_locations": uncertain
    }

def merge_location_results(spacy_results, regex_results):
    """Merge results from spaCy and regex extraction"""
    all_districts = list(set(spacy_results["districts"] + regex_results["districts"]))
    all_upazilas = list(set(spacy_results["upazilas"] + regex_results["upazilas"]))
    
    return {
        "districts": all_districts,
        "upazilas": all_upazilas,
        "uncertain_locations": spacy_results["uncertain_locations"]
    }

def extract_event_date(dates, text):
    """Try to identify the main event date (not publication date)"""
    # Look for phrases like "on August 21", "since August 20", etc.
    event_date_pattern = r'(?:on|since|from|during)\s+([A-Z][a-z]+\s+\d{1,2}(?:,?\s+\d{4})?)'
    event_dates = re.findall(event_date_pattern, text)
    
    if event_dates:
        return event_dates[0]
    elif dates:
        # If no clear event date, return the first extracted date as a guess
        return dates[0] + " (estimated)"
    return "not clearly mentioned"

def process_article(text, filename):
    """Main processing function"""
    doc = nlp(text)
    
    # Extract entities using spaCy
    locations = []
    dates = []
    
    for ent in doc.ents:
        if ent.label_ == "GPE":
            locations.append(ent.text)
        elif ent.label_ == "DATE":
            dates.append(ent.text)
    
    # Extract locations using regex (to catch Bangladesh-specific places)
    regex_locations = extract_locations_with_regex(text)
    
    # Categorize spaCy-detected locations
    spacy_categorized = categorize_locations_by_context(doc, locations)
    
    # Merge both results
    final_locations = merge_location_results(spacy_categorized, regex_locations)
    
    # Extract other data
    disaster_type = extract_disaster_type(text)
    categorized_numbers = extract_categorized_numbers(text)
    event_date = extract_event_date(dates, text)
    
    # Display results
    print("\n" + "="*60)
    print(f"Article: {filename}")
    print("="*60)
    print(f"Disaster Type: {disaster_type}")
    print(f"\nEvent Date: {event_date}")
    print(f"\nLocations:")
    print(f"  Districts: {final_locations['districts']}")
    print(f"  Upazilas:  {final_locations['upazilas']}")
    print(f"  Uncertain: {final_locations['uncertain_locations']}")
    print(f"\nCasualties & Impact:")
    print(f"  Deaths:   {categorized_numbers['deaths']}")
    print(f"  Injured:  {categorized_numbers['injured']}")
    print(f"  Affected: {categorized_numbers['affected']}")
    print("="*60 + "\n")

def main():
    """Process all articles in the raw_articles directory"""
    articles_dir = "data/raw_articles"
    
    if not os.path.exists(articles_dir):
        print(f"Error: Directory '{articles_dir}' not found!")
        return
    
    txt_files = [f for f in os.listdir(articles_dir) if f.endswith(".txt")]
    
    if not txt_files:
        print(f"No .txt files found in '{articles_dir}'")
        return
    
    print(f"Found {len(txt_files)} article(s) to process...\n")
    
    for filename in txt_files:
        filepath = os.path.join(articles_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            process_article(text, filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}\n")

if __name__ == "__main__":
    main()