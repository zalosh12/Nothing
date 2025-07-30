class DataSplitter:
    def __init__(self, df, class_column, test_frac=0.3,random_state=33):
        self.df = df
        self.class_column = class_column
        self.test_frac = test_frac
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.split()

    def split(self):
        df_shuffled = self.df.sample(frac=1,random_state=self.random_state).reset_index(drop=True)
        split_index = int(len(df_shuffled) * (1 - self.test_frac))

        train_df = df_shuffled[:split_index]
        test_df = df_shuffled[split_index:]

        self.X_train = train_df.drop(self.class_column, axis=1)
        self.y_train = train_df[self.class_column]
        self.X_test = test_df.drop(self.class_column, axis=1)
        self.y_test = test_df[self.class_column]