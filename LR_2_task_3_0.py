from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + "\n...")

print("Назви відповідей: {}".format(iris_dataset['target_names']))

print("Назва ознак: \n{}".format(iris_dataset['feature_names']))

print("Тип масиву data: {}".format(type(iris_dataset['data'])))

print("Форма масиву data: {}".format(iris_dataset['data'].shape))

print(iris_dataset['data'][0])
print(iris_dataset['data'][1])
print(iris_dataset['data'][2])
print(iris_dataset['data'][3])
print(iris_dataset['data'][4])

print("Тип масиву target: {}".format(type(iris_dataset['target'])))

print("Відповіді: \n{}".format(iris_dataset['target']))