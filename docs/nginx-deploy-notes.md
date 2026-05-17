# Notes de déploiement nginx

## Problème : POSTs qui hangent 10s puis timeout

**Symptôme** : les requêtes HTTP normales (`POST /api/quiz/score`, etc.) timeout
au bout de 10s côté client alors que Flask répond instantanément quand on tape
directement sur `http://127.0.0.1:5000`.

**Cause** : la config nginx d'origine forçait `Connection: upgrade` sur toutes
les requêtes (config classique pour Socket.IO) :

```nginx
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";   # ← problème
```

Pour une requête HTTP "normale" qui n'a pas d'en-tête `Upgrade:`, nginx envoie
quand même `Connection: upgrade` à Flask, qui attend l'upgrade puis timeout.

**Fix** : un `map` conditionnel qui n'utilise `upgrade` que si le client a
vraiment envoyé un `Upgrade:` :

`/etc/nginx/conf.d/hmh-upgrade-map.conf` :
```nginx
map $http_upgrade $hmh_connection_upgrade {
    default upgrade;
    ""      close;
}
```

Et dans `/etc/nginx/sites-available/hmh` :
```nginx
proxy_set_header Connection $hmh_connection_upgrade;   # au lieu de "upgrade"
```

Puis `sudo nginx -t && sudo systemctl reload nginx`.

Avec ça :
- Requête WebSocket (Socket.IO upgrade) → `Connection: upgrade` → OK
- Requête HTTP normale → `Connection: close` → OK, plus de timeout
