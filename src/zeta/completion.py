"""Static shell completion scripts for the zeta CLI."""

from __future__ import annotations

_INSTALL_COMMENT = "# install: zeta completion {shell} > {destination}\n"


def zsh_script() -> str:
    return _INSTALL_COMMENT.format(shell="zsh", destination="~/.zsh/completions/_zeta") + r'''#compdef zeta

_zeta() {
    local context state line command_index command_name token
    local -a commands session_verbs automation_verbs original_words
    typeset -A opt_args
    commands=(login serve session automation completion)
    session_verbs=(list rename delete export)
    automation_verbs=(list show approve disable import daemon)
    original_words=("${words[@]}")
    _arguments -C \
        '(-h --help)'{-h,--help}'[show help]' \
        '(-c --continue --resume --no-session)'{-c,--continue}'[resume the most recent session]' \
        '(-c --continue --resume --no-session)--resume=[resume a session]:session id:' \
        '(-c --continue --resume --no-session)--no-session[run without persistence]' \
        '--provider=[completion provider]:provider:(fake claude codex)' \
        '--model=[provider model override]:model:' \
        '--force-provider[allow provider or model overrides during resume]' \
        '--verbose[show raw stream events]' \
        '--yolo[auto-approve every tool call]' \
        '--no-yolo[force prompts even when settings enable yolo]' \
        '--token-budget=[context token budget]:tokens:' \
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
            command_index=2
            command_name=''
            while (( command_index <= $#original_words )); do
                token=$original_words[command_index]
                case $token in
                    --provider|--model|--resume|--token-budget|--max-turns|--format|--system-prompt|--append-system-prompt|--socket|--port|--cwd|-p|--print)
                        (( command_index += 2 ))
                        ;;
                    --provider=*|--model=*|--resume=*|--token-budget=*|--max-turns=*|--format=*|--system-prompt=*|--append-system-prompt=*|--socket=*|--port=*|--cwd=*|-p*)
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
            case $command_name in
                login)
                    _arguments '--provider=[OAuth provider]:provider:(anthropic codex)'
                    ;;
                serve)
                    _arguments '--socket=[Unix socket path]:path:' '--port=[localhost TCP port]:port:' '--provider=[completion provider]:provider:(fake claude codex)' '--model=[provider model]:model:' '--cwd=[working directory]:directory:_directories'
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
                        *) _describe 'verb' session_verbs ;;
                    esac
                    ;;
                automation)
                    case ${original_words[command_index+1]} in
                        list|daemon) _message 'no arguments' ;;
                        show|approve|disable) _arguments '1:name:' ;;
                        import) _arguments '1:JSON path:_files' ;;
                        *) _describe 'verb' automation_verbs ;;
                    esac
                    ;;
            esac
            ;;
    esac
}

compdef _zeta zeta
'''


def bash_script() -> str:
    return _INSTALL_COMMENT.format(shell="bash", destination="~/.local/share/bash-completion/completions/zeta") + r'''_zeta_completions() {
    local cur command verb token
    local command_index=0 index=1
    cur="${COMP_WORDS[COMP_CWORD]}"
    command=""
    verb=""
    while (( index < COMP_CWORD )); do
        token="${COMP_WORDS[index]}"
        case "$token" in
            --provider|--model|--resume|--token-budget|--max-turns|--format|--system-prompt|--append-system-prompt|--socket|--port|--cwd|-p|--print)
                (( index += 2 ))
                ;;
            --provider=*|--model=*|--resume=*|--token-budget=*|--max-turns=*|--format=*|--system-prompt=*|--append-system-prompt=*|--socket=*|--port=*|--cwd=*|-p*)
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
    local top_flags="-h --help --provider --model --continue -c --resume --no-session --force-provider --verbose --yolo --no-yolo --token-budget --max-turns --print -p --format --system-prompt --append-system-prompt --socket --port --cwd"
    local commands="login serve session automation completion"

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
            COMPREPLY=( $(compgen -W "--socket --port --provider --model --cwd" -- "$cur") )
            ;;
        completion)
            COMPREPLY=( $(compgen -W "zsh bash" -- "$cur") )
            ;;
        session)
            if (( COMP_CWORD <= command_index + 1 )); then
                COMPREPLY=( $(compgen -W "list rename delete export" -- "$cur") )
            else
                case "$verb" in
                    delete) COMPREPLY=( $(compgen -W "--force" -- "$cur") ) ;;
                    export) COMPREPLY=( $(compgen -W "--out" -- "$cur") ) ;;
                esac
            fi
            ;;
        automation)
            if (( COMP_CWORD <= command_index + 1 )); then
                COMPREPLY=( $(compgen -W "list show approve disable import daemon" -- "$cur") )
            fi
            ;;
        *)
            COMPREPLY=( $(compgen -W "$top_flags $commands" -- "$cur") )
            ;;
    esac
}

complete -F _zeta_completions zeta
'''


def completion_script(shell: str) -> str:
    """Return the deterministic completion script for ``shell``."""

    if shell == "zsh":
        return zsh_script()
    if shell == "bash":
        return bash_script()
    raise ValueError(f"unsupported shell: {shell}")


__all__ = ["bash_script", "completion_script", "zsh_script"]
