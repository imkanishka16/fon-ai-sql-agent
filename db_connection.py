import mysql.connector

config = {
  'user': 'root',
  'password': 'root',
  'host': '127.0.0.1',
  'port': 8889,
  'database': 'mydatabase',
  'raise_on_warnings': True
}

cnx = mysql.connector.connect(**config)

cursor = cnx.cursor(dictionary=True)

cursor.execute('SELECT `Country` FROM `PCIT`')

results = cursor.fetchall()

for row in results:
  country = row['Country']
  
  print( '%s' % (country))

cnx.close()
