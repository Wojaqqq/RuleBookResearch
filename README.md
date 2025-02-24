# RuleBookReaserch

To test single endpoint with POSTMAN do following:

```console
curl -X POST "http://127.0.0.1:5000/chat" \
     -H "Content-Type: application/json" \
     -d '{"game": "Monopoly", "query": "How much money do I have at the beginning of the game?"}'
```