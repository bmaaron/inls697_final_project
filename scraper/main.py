from src.scraper import scrape_info
from src.active_link_check import check_links, save_to_csv
from src.converter import convert_to_csv

def main():
    operation = input("Enter Website Address:")

    start_id = int(input("Please enter ID start #: "))
    end_id = int(input("Please enter ID end #: "))
    
    # Generate the list of codes from start_id to end_id inclusive
    customer_codes = list(range(start_id, end_id + 1))

    base_link = operation
    bearer_token = input("Please enter your bearer token: ")

    if operation == '1':
        # Check the links and get the statuses
        link_status_list = check_links(customer_codes, base_link, bearer_token)
        
        # Specify the CSV file name
        csv_filename = f"data/active_links/link_status_{start_id}_{end_id}.csv"

        # Save the link statuses to a CSV file
        save_to_csv(link_status_list, csv_filename)
    
    elif operation == '2':
        raw_json_path = f'data/json_data/raw_agency_data_id_no_{start_id}_to_{end_id}.json'
        output_csv_path = f'data/agency_data/agency_data_id_no_{start_id}_to_{end_id}.csv'

        # Scrape the information and save it to a JSON file
        scrape_info(customer_codes, raw_json_path, bearer_token)
        
        # Convert the JSON file to a CSV file
        convert_to_csv(raw_json_path, output_csv_path)
    

    elif operation == '3':
        raw_json_path = f'data/json_data/raw_agency_data_id_no_{start_id}_to_{end_id}.json'
        output_csv_path = f'data/agency_data/agency_data_id_no_{start_id}_to_{end_id}.csv'
        
        convert_to_csv(raw_json_path, output_csv_path)
   
    else:
        print("Invalid operation selected")


if __name__ == "__main__":
    main()
