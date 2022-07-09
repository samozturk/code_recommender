import sqlite3
import csv

# Connect to DB
conn = sqlite3.connect('snippets.db')

#id, snippet, language, repo_file_name, github_repo_url, license ,commit_hash 
# ,starting_line_number ,chunk_size ,UNIQUE(commit_hash, repo_file_name, 
# github_repo_url, chunk_size, starting_line_number)

# Fields to select
header = ['snippet', 'language']

cursor = conn.execute('''SELECT * FROM snippets 
WHERE "language"="Go" AND "snippet" NOT LIKE '%/%'
''')

with open('raw.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in cursor:
        writer.writerow([row[1], row[2]])