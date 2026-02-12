# linear-regression-
Predictive model to predict medical bill for patience based on input data
path = '/content/insurance.csv'
df = pd.read_csv(Path)

print("--- Data Snapshot ---")

print(df.head())
print("\n")

plt.figure(figsize=(10, 6))

sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')

plt.title('Missing Values Heatmap (Yellow lines = Missing Data)')
plt.savefig('missing_values_heatmap.png')
plt.show()
print("--- Missing Values Count ---")

print(df.isnull().sum())
print("\n")

categorical_columns = ['sex', 'smoker', 'region']

df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print("--- Data After Encoding (Text -> Numbers) ---")
print(df_encoded.head())
print("\n")

X = df_encoded.drop('charges', axis=1)

y = df_encoded['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

print("...Model Trained Successfully...\n")

predictions = model.predict(X_test)


r2 = r2_score(y_test, predictions)


print("--- Model Performance ---")
print(f"Model Accuracy (R^2 Score): {r2:.2f}")


def predict_insurance_cost():
    print("\n--- Medical Bill Predictor ---")
    print("Please enter the following details:")


    try:
        age = int(input("Age: "))
        bmi = float(input("BMI (e.g., 25.5): "))
        children = int(input("Number of Children: "))
        smoker = input("Smoker? (yes/no): ").lower()
        sex = input("Sex (male/female): ").lower()
        region = input("Region (southwest/southeast/northwest/northeast): ").lower()


        user_data = {
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex': [sex],
            'smoker': [smoker],
            'region': [region]
        }


        new_df = pd.get_dummies(pd.DataFrame(user_data), drop_first=True)
        new_df_aligned = new_df.reindex(columns=X.columns, fill_value=0)

        predicted_cost = model.predict(new_df_aligned)

        print(f"\nEstimated Insurance Bill: ${predicted_cost[0]:,.2f}")

    except ValueError:
        print("Invalid input! Please enter numbers for Age, BMI, and Children.")

predict_insurance_cost()
