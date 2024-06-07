import pandas as pd

# Load the CSV file
df = pd.read_csv('ProvModelInputData.csv')

# Drop first row (it has NA values and maintain biannual cycles)
df = df.drop(0)

# Convert the date column to the correct SQL format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

# Save the processed CSV
df.to_csv('WaterData.csv', index=False)


# Get the column names
columns = df.columns
dtypes = df.dtypes

# Initialize the CREATE TABLE statement
create_table_sql = "CREATE TABLE mytable (\n"

# Add columns definitions
for col, dtype in zip(columns, dtypes):
    if dtype == 'object':
        create_table_sql += f"    {col} DATE,\n"
    elif dtype == 'int64':
        create_table_sql += f"    {col} INT,\n"
    elif dtype == 'float64':
        create_table_sql += f"    {col} FLOAT,\n"
    else:
        create_table_sql += f"    {col} VARCHAR(255),\n"  # Default to VARCHAR for other types

# Remove the last comma and newline, then close the statement
create_table_sql = create_table_sql.rstrip(',\n') + "\n);"

print(create_table_sql)
