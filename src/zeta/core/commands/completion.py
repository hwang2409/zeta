"""Static shell completion scripts for the zeta CLI."""

from __future__ import annotations

_INSTALL_COMMENT = "# install: zeta completion {shell} > {destination}\n"


def zsh_script() -> str:
    return (
        _INSTALL_COMMENT.format(shell="zsh", destination="~/.zsh/completions/_zeta")
        + r"""#compdef zeta

_zeta() {
    local context state line command_index command_name token
    local -a commands session_verbs automation_verbs project_verbs project_index_verbs webhook_verbs mcp_verbs original_words
    typeset -A opt_args
    commands=(login serve session project inbox panel automation mcp stalls completion)
    session_verbs=(list rename delete export stats push pull)
    automation_verbs=(list show approve disable import daemon webhook)
    project_verbs=(create init discover list show memory index)
    project_index_verbs=(status rebuild search)
    webhook_verbs=(url show-secret rotate-secret rotate-url)
    mcp_verbs=(add list show remove enable disable test login logout trust untrust)
    original_words=("${words[@]}")
    command_index=2
    command_name=''
    while (( command_index <= $#original_words )); do
        token=$original_words[command_index]
        case $token in
            --provider|--model|--resume|--token-budget|--compaction|--tools|--disallowed-tools|--max-turns|--format|--system-prompt|--append-system-prompt|-p|--print)
                (( command_index += 2 ))
                ;;
            --provider=*|--model=*|--resume=*|--token-budget=*|--compaction=*|--tools=*|--disallowed-tools=*|--max-turns=*|--format=*|--system-prompt=*|--append-system-prompt=*|-p*)
                (( command_index++ ))
                ;;
            --)
                (( command_index++ ))
                command_name=${original_words[command_index]}
                break
                ;;
            -*)
                (( command_index++ ))
                ;;
            *)
                command_name=$token
                break
                ;;
        esac
    done
    if [[ $command_name == automation && ${original_words[command_index+1]} == daemon && ${original_words[CURRENT]} == --* ]]; then
        compadd -- --webhook-host --webhook-port --allow-non-loopback
        return
    fi
    _arguments -C \
        '(-h --help)'{-h,--help}'[show help]' \
        '(-c --continue --resume --no-session)'{-c,--continue}'[resume the most recent session]' \
        '(-c --continue --resume --no-session)--resume=[resume a session]:session id:' \
        '(-c --continue --resume --no-session)--no-session[run without persistence]' \
        '--provider=[completion provider]:provider:(fake claude codex ollama)' \
        '--model=[provider model override]:model:' \
        '--force-provider[allow provider or model overrides during resume]' \
        '--verbose[show raw stream events]' \
        '--yolo[auto-approve every tool call]' \
        '--no-yolo[force prompts even when settings enable yolo]' \
        '--auto-memory[enable automatic project memory]' \
        '--no-auto-memory[disable automatic project memory]' \
        '--token-budget=[context token budget]:tokens:' \
        '--compaction=[context compaction mode]:mode:(summary evict)' \
        '--tools=[tool allowlist]:pattern list:' \
        '--disallowed-tools=[tool denylist]:pattern list:' \
        '--require-tools[fail when exact allowlisted tools are unavailable]' \
        '--allow-hooks[run trusted hooks in restricted sessions]' \
        '--max-turns=[tool-use loop turn cap]:turns:' \
        '(-p --print)'{-p,--print}'[run one headless turn]:prompt:' \
        '--format=[headless output format]:format:(text json)' \
        '--system-prompt=[replace the built-in system prompt]:text or @file:' \
        '--append-system-prompt=[append to the default system prompt]:text or @file:' \
        '1:command:->command' \
        '*::argument:->argument'

    case $state in
        command)
            _describe 'command' commands
            ;;
        argument)
            case $command_name in
                login)
                    _arguments '--provider=[OAuth provider]:provider:(anthropic codex)'
                    ;;
                serve)
                    _arguments '--socket=[Unix socket path]:path:' '--port=[localhost TCP port]:port:' '--provider=[completion provider]:provider:(fake claude codex ollama)' '--model=[provider model]:model:' '--cwd=[working directory]:directory:_directories' '--tools=[tool allowlist]:pattern list:' '--disallowed-tools=[tool denylist]:pattern list:' '--require-tools[fail when exact allowlisted tools are unavailable]' '--allow-hooks[run trusted hooks in restricted sessions]' '--auto-memory[enable automatic project memory]' '--no-auto-memory[disable automatic project memory]'
                    ;;
                stalls)
                    _arguments '--top=[top stacks]:count:' '--json[print JSON]'
                    ;;
                completion)
                    _arguments '1:shell:(zsh bash)'
                    ;;
                session)
                    case ${original_words[command_index+1]} in
                        list) _message 'no arguments' ;;
                        rename) _arguments '1:session id:' '2:display name:' ;;
                        delete) _arguments '--force[skip confirmation]' '1:session id:' ;;
                        export) _arguments '--out=[output file]:file:_files' '1:session id:' ;;
                        stats) _arguments '--compaction[compaction report]' '--since=[window or date]:since:' '--session=[session id or prefix]:session id:' '--top=[top sessions]:count:' '--json[print JSON]' ;;
                        push) _arguments '--force[replace divergent remote state]' '--remote-home=[remote ZETA_HOME]:directory:' '1:host:' '2:session id:' ;;
                        pull) _arguments '--force[replace divergent local state]' '--remote-home=[remote ZETA_HOME]:directory:' '--cwd=[mapped working directory]:directory:_directories' '1:host:' '2:session id:' ;;
                        *) _describe 'verb' session_verbs ;;
                    esac
                    ;;
                project)
                    case ${original_words[command_index+1]} in
                        create) _arguments '--scope=[project scope]:scope:' '--canonical-integration-root=[directory]:directory:_directories' '1:name:' ;;
                        init) _arguments '--name=[project name]:name:' '--scope=[project scope]:scope:' '1:directory:_directories' ;;
                        discover) _arguments '1:directory:_directories' ;;
                        memory) _arguments '--project=[project for sync]:project:' '--remote-home=[remote ZETA_HOME]:directory:' '--accept=[conflict side]:side:(local remote)' '1:project or push/pull/resolve:' '2:remote host:' '--set=[memory file]:file:' '2:content:' '--from-file=[memory file]:file:' '2:path:_files' ;;
                        index) _arguments '1:project:' '2:action:(status rebuild search)' '--limit=[result limit]:count:' '*:query:' ;;
                        show) _arguments '1:project:' ;;
                        list) _message 'no arguments' ;;
                        *) _describe 'verb' project_verbs ;;
                    esac
                    ;;
                inbox)
                    _arguments '--project=[project name or ID]:project:'
                    ;;
                panel)
                    _arguments '--list[print a plain text summary]'
                    ;;
                automation)
                    case ${original_words[command_index+1]} in
                        list) _message 'no arguments' ;;
                        daemon) _arguments '--webhook-host=[webhook bind host]:host:' '--webhook-port=[webhook bind port]:port:' '--allow-non-loopback[allow a non-loopback webhook bind]' ;;
                        show|approve|disable) _arguments '1:name:' ;;
                        import) _arguments '*:JSON path:_files' ;;
                        webhook)
                            case ${original_words[command_index+2]} in
                                url|show-secret|rotate-secret|rotate-url) _arguments '1:name:' ;;
                                *) _describe 'webhook verb' webhook_verbs ;;
                            esac
                            ;;
                        *) _describe 'verb' automation_verbs ;;
                    esac
                    ;;
                mcp)
                    case ${original_words[command_index+1]} in
                        add) _arguments '--scope=[configuration scope]:scope:(user project)' '--url=[HTTP server URL]:URL:' '--oauth[use OAuth]' '*--env=[environment reference]:KEY=VALUE:' '*--header=[HTTP header reference]:HEADER=VALUE:' '1:name:' '*:server command:' ;;
                        list) _arguments '--scope=[configuration scope]:scope:(user project effective)' '--json[emit JSON]' ;;
                        show) _arguments '--scope=[configuration scope]:scope:(user project effective)' '--json[emit JSON]' '1:name:' ;;
                        remove|enable|disable) _arguments '--scope=[configuration scope]:scope:(user project)' '1:name:' ;;
                        test|login|logout) _arguments '--scope=[configuration scope]:scope:(user project effective)' '1:name:' ;;
                        trust|untrust) _arguments '1:name:' ;;
                        *) _describe 'verb' mcp_verbs ;;
                    esac
                    ;;
            esac
            ;;
    esac
}

compdef _zeta zeta
"""
    )


def bash_script() -> str:
    return (
        _INSTALL_COMMENT.format(
            shell="bash", destination="~/.local/share/bash-completion/completions/zeta"
        )
        + r"""_zeta_completions() {
    local cur command verb token
    local command_index=0 index=1
    cur="${COMP_WORDS[COMP_CWORD]}"
    command=""
    verb=""
    while (( index < COMP_CWORD )); do
        token="${COMP_WORDS[index]}"
        case "$token" in
            --provider|--model|--resume|--token-budget|--compaction|--tools|--disallowed-tools|--max-turns|--format|--system-prompt|--append-system-prompt|--socket|--port|--cwd|-p|--print)
                if [[ "${COMP_WORDS[index+1]:-}" == "=" ]]; then
                    (( index += 3 ))
                else
                    (( index += 2 ))
                fi
                ;;
            --provider=*|--model=*|--resume=*|--token-budget=*|--compaction=*|--tools=*|--disallowed-tools=*|--max-turns=*|--format=*|--system-prompt=*|--append-system-prompt=*|--socket=*|--port=*|--cwd=*|-p*)
                (( index++ ))
                ;;
            --)
                (( index++ ))
                if (( index < COMP_CWORD )); then
                    command="${COMP_WORDS[index]}"
                    command_index=$index
                fi
                break
                ;;
            -*)
                (( index++ ))
                ;;
            *)
                command="$token"
                command_index=$index
                break
                ;;
        esac
    done
    if (( command_index > 0 && COMP_CWORD > command_index + 1 )); then
        verb="${COMP_WORDS[command_index+1]}"
    fi
    local top_flags="-h --help --provider --model --continue -c --resume --no-session --force-provider --verbose --yolo --no-yolo --auto-memory --no-auto-memory --token-budget --compaction --tools --disallowed-tools --require-tools --allow-hooks --max-turns --print -p --format --system-prompt --append-system-prompt"
    local commands="login serve session project inbox panel automation mcp stalls completion"

    if (( command_index == 0 )); then
        if [[ "$cur" == -* ]]; then
            COMPREPLY=( $(compgen -W "$top_flags" -- "$cur") )
        else
            COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
        fi
        return
    fi

    case "$command" in
        login)
            COMPREPLY=( $(compgen -W "--provider" -- "$cur") )
            ;;
        serve)
            COMPREPLY=( $(compgen -W "--socket --port --provider --model --cwd --tools --disallowed-tools --require-tools --allow-hooks --auto-memory --no-auto-memory" -- "$cur") )
            ;;
        stalls)
            COMPREPLY=( $(compgen -W "--top --json" -- "$cur") )
            ;;
        completion)
            COMPREPLY=( $(compgen -W "zsh bash" -- "$cur") )
            ;;
        session)
            if (( COMP_CWORD <= command_index + 1 )); then
                COMPREPLY=( $(compgen -W "list rename delete export stats push pull" -- "$cur") )
            else
                case "$verb" in
                    delete) COMPREPLY=( $(compgen -W "--force" -- "$cur") ) ;;
                    export) COMPREPLY=( $(compgen -W "--out" -- "$cur") ) ;;
                    stats) COMPREPLY=( $(compgen -W "--compaction --since --session --top --json" -- "$cur") ) ;;
                    push) COMPREPLY=( $(compgen -W "--force --remote-home" -- "$cur") ) ;;
                    pull) COMPREPLY=( $(compgen -W "--force --remote-home --cwd" -- "$cur") ) ;;
                esac
            fi
            ;;
        project)
            if (( COMP_CWORD <= command_index + 1 )); then
                COMPREPLY=( $(compgen -W "create init discover list show memory index --canonical-integration-root --name --scope --set --from-file --project --remote-home --accept" -- "$cur") )
            else
                case "$verb" in
                    create) COMPREPLY=( $(compgen -W "--scope --canonical-integration-root" -- "$cur") ) ;;
                    init) COMPREPLY=( $(compgen -W "--name --scope" -- "$cur") ) ;;
                    memory) COMPREPLY=( $(compgen -W "push pull resolve --set --from-file --project --remote-home --accept" -- "$cur") ) ;;
                    index) COMPREPLY=( $(compgen -W "status rebuild search --limit" -- "$cur") ) ;;
                esac
            fi
            ;;
        inbox)
            COMPREPLY=( $(compgen -W "--project" -- "$cur") )
            ;;
        panel)
            COMPREPLY=( $(compgen -W "--list" -- "$cur") )
            ;;
        automation)
            if (( COMP_CWORD <= command_index + 1 )); then
                COMPREPLY=( $(compgen -W "list show approve disable import daemon webhook" -- "$cur") )
            elif [[ "$verb" == "webhook" && $COMP_CWORD -le $((command_index + 2)) ]]; then
                COMPREPLY=( $(compgen -W "url show-secret rotate-secret rotate-url" -- "$cur") )
            elif [[ "$verb" == "daemon" ]]; then
                COMPREPLY=( $(compgen -W "--webhook-host --webhook-port --allow-non-loopback" -- "$cur") )
            fi
            ;;
        mcp)
            if (( COMP_CWORD <= command_index + 1 )); then
                COMPREPLY=( $(compgen -W "add list show remove enable disable test login logout trust untrust" -- "$cur") )
            else
                case "$verb" in
                    add) COMPREPLY=( $(compgen -W "--scope --url --oauth --env --header" -- "$cur") ) ;;
                    list|show) COMPREPLY=( $(compgen -W "--scope --json" -- "$cur") ) ;;
                    remove|enable|disable|test|login|logout) COMPREPLY=( $(compgen -W "--scope" -- "$cur") ) ;;
                esac
            fi
            ;;
        *)
            COMPREPLY=( $(compgen -W "$top_flags $commands" -- "$cur") )
            ;;
    esac
}

complete -F _zeta_completions zeta
"""
    )


def completion_script(shell: str) -> str:
    """Return the deterministic completion script for ``shell``."""

    if shell == "zsh":
        return zsh_script()
    if shell == "bash":
        return bash_script()
    raise ValueError(f"unsupported shell: {shell}")


__all__ = ["bash_script", "completion_script", "zsh_script"]
