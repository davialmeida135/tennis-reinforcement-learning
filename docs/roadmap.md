# A fazer
## Parser inicial de pontos
- Consertar o primeiro parser pra colocar 0 em vez de unknown nas direções e ignorar golpes que não tem direção.
- Usar direções só 1,2,3 desde o começo.
- Remover todos os unknown
- Erros terem direção(?)

## Construção do grafo de transição
- Organizar o código depois de consertar o parser de pontos

## Ambiente
- Integrar Engine de partida de tenis
1. Ambiente recebe ação
1. Ambiente gera próximas duas ações
1. Se a primeira for erro, ponto pro pc -> Verifica placar -> Se for a vez do pc, gera mais ações
1. Se a segunda for erro, ponto pro jogador -> Verifica placar -> Se for a vez do pc, gera mais 2 ações

