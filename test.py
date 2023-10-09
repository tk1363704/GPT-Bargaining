import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Create a Counter object
counter = Counter()

list_ = ['I can\'t go that low, but I can offer it to you for $18.', 'I understand your offer, but the lowest I can go is $17.', 'Great! Deal! The balloon is yours for $17. Enjoy!', 'Hi, this is a good balloon and its price is $20']
for x in list_:
    counter.update(x.strip().split())

# # Convert the list of words into a single string
# text = " ".join(word_list)

max_words = 30
print(counter)

# # Generate the word cloud
# wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate(text)
wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate_from_frequencies(counter)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # Turn off the axis
plt.show()

# Save the word cloud image as a PNG file
wordcloud.to_file("wordcloud.png")

# list_ = [1,2,3,4,6]
# string_ = ' '.join([str(x) for x in list_]).strip()
# print(string_)


