Data API: https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script

Data fetching cURL command:  curl -X POST "https://www.daggegevens.knmi.nl/klimatologie/daggegevens" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data "start=20220101&end=20231231&stns=240&vars=ALL&fmt=json" \
  -o amsterdam_2022_2023.json
