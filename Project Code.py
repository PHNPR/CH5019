from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def similarity(answers):
	vectorizer = TfidfVectorizer()
	vector = vectorizer.fit_transform(answers)
	cosine_similarities = cosine_similarity(vector[0],vector[1])

	percentage = round(float(cosine_similarities) * 100 , 4)
	print(percentage)

correct_answer = input("Enter the correct answer : ")
student_answer = input("Enter the student's answer : ")
answers = [correct_answer , student_answer]

similarity(answers)
