import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# # Create a Counter object
# counter = Counter()
#
# list_ = ['I can\'t go that low, but I can offer it to you for $18.', 'I understand your offer, but the lowest I can go is $17.', 'Great! Deal! The balloon is yours for $17. Enjoy!', 'Hi, this is a good balloon and its price is $20']
# for x in list_:
#     counter.update(x.strip().split())
#
# # # Convert the list of words into a single string
# # text = " ".join(word_list)
#
# max_words = 30
# print(counter)
#
# # # Generate the word cloud
# # wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate(text)
# wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate_from_frequencies(counter)
#
# # Display the word cloud using matplotlib
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")  # Turn off the axis
# plt.show()
#
# # Save the word cloud image as a PNG file
# wordcloud.to_file("wordcloud.png")
#
# # list_ = [1,2,3,4,6]
# # string_ = ' '.join([str(x) for x in list_]).strip()
# # print(string_)


def extract():
    content = ""
    with open('test.txt', 'r') as file:
        # Read all lines
        lines = file.readlines()
    for line in lines:
        content += line
    content = content.strip()
    lists = [x.strip() for x in content.split('\n') if x.strip()!='']
    print(lists)

    level_2_result, level_1_result = "", ""
    for i, item in enumerate(lists):
        if '2. conversion' in item.lower() and i < len(lists)-1:
            level_2_result = lists[i+1]
        if 'further conversion' in item.lower() and i < len(lists)-1:
            level_1_result = lists[i+1]
    return level_2_result, level_1_result


def main():
    # print(extract())
    str_ = " <s> System: 你是一个对话助手，你的任务是改写对话中最后一句话，使其符合社会规范。\n在谈判游戏中，谈判的对话如下：\n人物A: 你好，请问贵公司有一种特殊的工业商品吗？\n人物B：你好，我们公司产能充足，能够以一次性的方式向贵司出售所需的工业商品。这种工业商品的单元价格是$50。\n人物A: 你会考虑以每单元$30的价格进行出售吗？\n人物B：很抱歉，我们不会接受低于$40的价格。\n在这个对话中，最后一句话\"很抱歉，我们不会接受低于$40的价格\"违反了社会规范，请对这句话进行改写:\n</s><s> Assistant: 很遗憾，我们的起始报价是您无法满足的需求价码。\n</s>"
    list_ = str_.split('<s>')
    print(list_)
    remedy_ = str_.split('<s>')[-1].replace('</s>', '').replace('Assistant:', '').replace('assistant:', '').strip()
    print(remedy_)


if __name__ == "__main__":
    main()
