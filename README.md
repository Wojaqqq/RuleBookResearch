# RuleBookReaserch

To test single endpoint with POSTMAN do following:

```console
curl -X POST http://localhost:5000/chat \
     -H "Content-Type: application/json" \
     -d '{
         "game": "monopoly",
         "query": "How much money do I have at the beginning of the game?",
         "fine_tuned": false
     }'
curl -X POST http://localhost:5000/chat \
     -H "Content-Type: application/json" \
     -d '{
         "game": "game_of_thrones",
         "query": "What can I do with raid token?",
         "fine_tuned": false
     }'
```