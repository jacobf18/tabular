# Multiple missingness types in a list

missingness_types=("mcar" "mar" "mnar")
columns=("5" "10" "20" "50")
rows=("5" "10" "20" "50")

for missingness_type in "${missingness_types[@]}"; do
    for column in "${columns[@]}"; do
        for row in "${rows[@]}"; do
            echo "Generating data for Missingness: ${missingness_type} Columns: ${column} Rows: ${row}"
            nohup sh generate_data.sh ${missingness_type} scm ${column} ${row} > ${missingness_type}_${column}_${row}.log 2>&1 &
        done
    done
done
