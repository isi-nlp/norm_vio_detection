import os

if __name__ == '__main__':
    command = f"curl -X POST -H 'Content-Type: application/json' -d '@api-test-data.json' http://localhost:5000/api"
    print(command)
    print('\n')
    os.system(command)
