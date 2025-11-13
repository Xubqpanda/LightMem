# Demonstration of list append vs append(copy) and clear()

buffer = []
segments = []

# Add an item and append the buffer reference
buffer.append({'role': 'user', 'content': "favorite ice cream flavor is pistachio dog ' s name is Rex"})
buffer.append({'role': 'assistant', 'content': 'Got it. Pistachio great choice'})

print('Before appending: buffer =', buffer)
segments.append(buffer)
print('After appending reference: segments =', segments)

# Clear buffer
buffer.clear()
print('\nAfter buffer.clear():')
print('buffer =', buffer)
print('segments =', segments)

# Now demonstrate appending a shallow copy instead
buffer.append({'role': 'user', 'content': "favorite ice cream flavor is pistachio dog ' s name is Rex"})
buffer.append({'role': 'assistant', 'content': 'Got it. Pistachio great choice'})

segments = []
segments.append(buffer.copy())
print('\nAfter appending buffer.copy(): segments =', segments)

# Clear buffer again
buffer.clear()
print('\nAfter buffer.clear() when we appended a copy:')
print('buffer =', buffer)
print('segments =', segments)
